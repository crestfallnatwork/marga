// SPDX-License-Identifier: GPL-2.0-only
// Author: Himanshu Gupta <himanshu3@iisc.ac.in>

// Shiloach-Vishkin-style connected components over a shrinking active list.
// The task stage finds both roots, culls resolved edges, compacts live (ru, rv)
// pairs for the next round, and stages (ru, rv) for the mesh stage; the mesh
// stage performs one conditional root hook per live pair.
//
// Round 0 reads the CSR directly: the source vertex of an edge is recovered
// from the row offsets by a per-group search plus monotone advance, so no
// per-edge source array has to be built. Later rounds work only on the active
// pair list.
//
// Each task group covers cGroupItems = WAVE_SIZE * cEdgesPerLane items: the
// per-item work is too small to amortise one dispatch per 32 items, so every
// lane handles cEdgesPerLane items and the group amortises the dispatch.

static const uint cMaxMeshDim   = 65535;
static const uint cEdgesPerLane = 8;
static const uint cGroupItems   = WAVE_SIZE * cEdgesPerLane;

T At<T>(T* ptr, uint64_t offset) {
    return *(T*)(uint64_t(ptr) + sizeof(T) * offset);
}
T* Splice<T>(T* ptr, uint64_t offset) {
    return (T*)(uint64_t(ptr) + sizeof(T) * offset);
}

// Path-halving find. The write is benign: any value read along the parent
// chain is an ancestor of x, so a racing write can only shorten the path.
uint FindRoot(uint32_t* parent, uint x) {
    uint p = At(parent, x);
    while (p != x) {
        uint gp = At(parent, p);
        if (gp == p) {
            return p;
        }
        *Splice(parent, x) = gp;
        x = gp;
        p = At(parent, x);
    }
    return x;
}

// Row containing edge eid in a CSR with `n` rows (upper-bound search).
uint FindRow(uint64_t* offsets, uint64_t n, uint64_t eid) {
    uint64_t low = 0;
    uint64_t high = n;
    while (low < high) {
        uint64_t mid = low + (high - low) / 2;
        if (At(offsets, mid) > eid) {
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    return uint(low - 1);
}

struct SharedCCPayload {
    uint ru[cGroupItems];
    uint rv[cGroupItems];
    uint offsets[cGroupItems + 1];
};
groupshared SharedCCPayload ccPayload;

struct CCIn {
    uint64_t* row_offsets;
    uint32_t* edge_lists;
    uint32_t* parent;
    // Active list: two u32 per entry (ru, rv).
    uint32_t* active;
    uint32_t* next_active;
    // counters[0] = successful hooks this round, counters[1] = next list count.
    uint32_t* counters;
    uint64_t num_vertices;
    uint64_t edge_start;
    uint64_t edge_num;
    // 0 = read the CSR (edge_start is an edge id), 1 = read the active list.
    uint32_t use_active;
    // 1 = symmetric input view: process each undirected edge once (u < v).
    uint32_t cull_upper;
    // Entries the next list can hold; live pairs beyond it are counted but
    // not stored, and the host rescans the CSR instead of using the list.
    uint64_t list_capacity;
};
// @bind(cc_task)
[[vk::push_constant]]
uniform CCIn ccIn;

[shader("amplification")]
[numthreads(WAVE_SIZE, 1, 1)]
void cc_task(
    uint3 gid : SV_GroupID,
    uint lid : SV_GroupIndex,
) {
    uint g = (gid.y * cMaxMeshDim) + gid.x;
    uint64_t group_base = uint64_t(g) * cGroupItems;

    // Round 0: the group's edges are consecutive, so one search finds the row
    // of the first edge and every item advances from there.
    uint u_start = 0;
    uint64_t eid_start = ccIn.edge_start + group_base;
    if (ccIn.use_active == 0 && WaveIsFirstLane() && eid_start < ccIn.edge_start + ccIn.edge_num) {
        u_start = FindRow(ccIn.row_offsets, ccIn.num_vertices, eid_start);
    }
    u_start = WaveReadLaneFirst(u_start);

    // Each lane's edge ids increase with i, so the row search advances
    // monotonically instead of restarting from the group's first row.
    uint u_cur = u_start;

    uint running = 0;
    [unroll]
    for (uint i = 0; i < cEdgesPerLane; ++i) {
        uint item = i * WAVE_SIZE + lid;
        uint64_t local = group_base + item;

        uint live = 0;
        uint ru   = 0;
        uint rv   = 0;

        if (local < ccIn.edge_num) {
            if (ccIn.use_active != 0) {
                uint64_t entry = (ccIn.edge_start + local) * 2;
                ru = At(ccIn.active, entry);
                rv = At(ccIn.active, entry + 1);
                ru = FindRoot(ccIn.parent, ru);
                rv = FindRoot(ccIn.parent, rv);
                live = (ru != rv) ? 1u : 0u;
            } else {
                uint64_t eid = ccIn.edge_start + local;
                uint u = u_cur;
                while (eid >= At(ccIn.row_offsets, uint64_t(u) + 1)) {
                    ++u;
                }
                u_cur = u;
                uint v = At(ccIn.edge_lists, eid);
                bool culled = (ccIn.cull_upper != 0) && (u > v);
                if (!culled) {
                    ru = FindRoot(ccIn.parent, u);
                    rv = FindRoot(ccIn.parent, v);
                    live = (ru != rv) ? 1u : 0u;
                }
            }
        }

        uint prefix = WavePrefixSum(live);
        uint total  = WaveActiveSum(live);

        ccPayload.ru[item]      = ru;
        ccPayload.rv[item]      = rv;
        ccPayload.offsets[item] = running + prefix;
        running += total;
    }

    if (lid == WAVE_SIZE - 1) {
        ccPayload.offsets[cGroupItems] = running;
    }

    GroupMemoryBarrierWithGroupSync();

    // One reservation per task group: a counter shared by every item would
    // serialise on a single address and dominate the round.
    uint base = 0;
    if (lid == 0) {
        uint previous = 0;
        InterlockedAdd(*Splice(ccIn.counters, 1), running, previous);
        base = previous;
    }
    base = WaveReadLaneFirst(base);

    [unroll]
    for (uint i = 0; i < cEdgesPerLane; ++i) {
        uint item = i * WAVE_SIZE + lid;
        uint begin = ccPayload.offsets[item];
        if (ccPayload.offsets[item + 1] > begin) {
            uint64_t dst = uint64_t(base) + begin;
            if (dst < ccIn.list_capacity) {
                *Splice(ccIn.next_active, dst * 2 + 0) = ccPayload.ru[item];
                *Splice(ccIn.next_active, dst * 2 + 1) = ccPayload.rv[item];
            }
        }
    }

    uint mesh_count = (running + 127) / 128;
    uint mesh_x     = min(mesh_count, cMaxMeshDim);
    uint mesh_y     = mesh_count > cMaxMeshDim ? ((mesh_count + cMaxMeshDim - 1) / cMaxMeshDim) : 1;

    DispatchMesh(mesh_x, mesh_y, 1, ccPayload);
}

[shader("mesh")]
[numthreads(128, 1, 1)]
[outputtopology("point")]
void cc_mesh(
    uint lid : SV_GroupIndex,
    uint3 gid : SV_GroupID,
    in payload SharedCCPayload inPayload,
    out vertices float4 vertices[1],
    out indices uint indices[1]
) {
    uint g      = (gid.y * cMaxMeshDim) + gid.x;
    uint thread = g * 128 + lid;
    uint total  = inPayload.offsets[cGroupItems];
    if (thread >= total) {
        SetMeshOutputCounts(0, 0);
        return;
    }

    // Map the global mesh thread to its live pair within this task group.
    uint item = 0;
    [unroll]
    for (uint step = (cGroupItems >> 1); step > 0; step >>= 1) {
        if (inPayload.offsets[item + step] <= thread) {
            item += step;
        }
    }

    uint ru = inPayload.ru[item];
    uint rv = inPayload.rv[item];

    // Hook the larger root under the smaller one. Every write goes from a
    // larger vertex to a smaller one, so parent chains stay acyclic even when
    // two lanes overwrite the same slot; a link that is lost or overwritten
    // is simply retried in the next round.
    if (ru < rv) {
        *Splice(ccIn.parent, rv) = ru;
    } else if (rv < ru) {
        *Splice(ccIn.parent, ru) = rv;
    }

    SetMeshOutputCounts(0, 0);
}

// One-time setup: parent[i] = i. Outside the round loop.
struct PrepareIn {
    uint32_t* parent;
    uint64_t  num_vertices;
    uint64_t  stride;
};
// @bind(prepare)
[[vk::push_constant]]
uniform PrepareIn prepareIn;

[shader("compute")]
[numthreads(256, 1, 1)]
void prepare(uint3 tid : SV_DispatchThreadID) {
    uint64_t i = uint64_t(tid.y) * prepareIn.stride + tid.x;
    if (i >= prepareIn.num_vertices) {
        return;
    }
    *Splice(prepareIn.parent, i) = uint32_t(i);
}
