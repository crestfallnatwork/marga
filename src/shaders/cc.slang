// SPDX-License-Identifier: GPL-2.0-only
// Author: Himanshu Gupta <himanshu3@iisc.ac.in>

// Shiloach-Vishkin-style connected components over a shrinking active edge
// list. The task stage finds both roots, culls resolved edges, compacts the
// live edge ids for the next round, and stages (ru, rv) per live edge; the
// mesh stage performs one conditional root hook per live edge.

static const uint cMaxMeshDim = 65535;

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

struct SharedCCPayload {
    uint ru[WAVE_SIZE];
    uint rv[WAVE_SIZE];
    uint offsets[WAVE_SIZE + 1];
};
groupshared SharedCCPayload ccPayload;

struct CCIn {
    uint32_t* edge_src;
    uint32_t* edge_lists;
    uint32_t* parent;
    uint32_t* active;
    uint32_t* next_active;
    // counters[0] = successful hooks this round, counters[1] = next active count.
    uint32_t* counters;
    uint64_t edge_start;
    uint64_t edge_num;
    // 0 = identity mapping from edge_start, 1 = read edge ids from `active`.
    uint32_t use_active;
    // 1 = symmetric input view: process each undirected edge once (u < v).
    uint32_t cull_upper;
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
    uint64_t local = uint64_t(g) * WAVE_SIZE + lid;

    uint live = 0;
    uint ru   = 0;
    uint rv   = 0;
    uint u    = 0;
    uint v    = 0;
    uint64_t eid = 0;

    if (local < ccIn.edge_num) {
        eid = (ccIn.use_active != 0)
                  ? uint64_t(At(ccIn.active, ccIn.edge_start + local))
                  : (ccIn.edge_start + local);
        u = At(ccIn.edge_src, eid);
        v = At(ccIn.edge_lists, eid);
        bool culled = (ccIn.cull_upper != 0) && (u > v);
        if (!culled) {
            ru = FindRoot(ccIn.parent, u);
            rv = FindRoot(ccIn.parent, v);
            live = (ru != rv) ? 1u : 0u;
        }
    }

    uint prefix = WavePrefixSum(live);
    uint total  = WaveActiveSum(live);

    uint base = 0;
    if (WaveIsFirstLane()) {
        uint previous = 0;
        InterlockedAdd(*Splice(ccIn.counters, 1), total, previous);
        base = previous;
    }
    base = WaveReadLaneFirst(base);

    if (live != 0) {
        *Splice(ccIn.next_active, uint64_t(base) + prefix) = uint32_t(eid);
    }

    ccPayload.ru[lid]      = ru;
    ccPayload.rv[lid]      = rv;
    ccPayload.offsets[lid] = prefix;
    if (lid == WAVE_SIZE - 1) {
        ccPayload.offsets[WAVE_SIZE] = total;
    }

    GroupMemoryBarrierWithGroupSync();

    uint mesh_count = (total + 127) / 128;
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
    uint total  = inPayload.offsets[WAVE_SIZE];
    if (thread >= total) {
        SetMeshOutputCounts(0, 0);
        return;
    }

    // Map the global mesh thread to its live edge within this task group.
    uint lane = 0;
    [unroll]
    for (uint step = (WAVE_SIZE >> 1); step > 0; step >>= 1) {
        if (inPayload.offsets[lane + step] <= thread) {
            lane += step;
        }
    }

    uint ru = inPayload.ru[lane];
    uint rv = inPayload.rv[lane];

    // Hook the larger root under the smaller one; only a live root can be
    // hooked, so a failed exchange is retried in the next round.
    uint hooked = 0;
    uint previous = 0;
    if (ru < rv) {
        InterlockedCompareExchange(*Splice(ccIn.parent, rv), rv, ru, previous);
        hooked = (previous == rv) ? 1u : 0u;
    } else if (rv < ru) {
        InterlockedCompareExchange(*Splice(ccIn.parent, ru), ru, rv, previous);
        hooked = (previous == ru) ? 1u : 0u;
    }

    if (hooked != 0) {
        InterlockedAdd(*Splice(ccIn.counters, 0), 1u);
    }

    SetMeshOutputCounts(0, 0);
}

// One-time setup: mode 0 writes parent[i] = i, mode 1 writes the source vertex
// of every edge from the CSR offsets. Both are outside the round loop.
struct PrepareIn {
    uint32_t* parent;
    uint32_t* edge_src;
    uint64_t* row_offsets;
    uint64_t num_vertices;
    uint64_t stride;
    uint32_t mode;
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
    if (prepareIn.mode == 0) {
        *Splice(prepareIn.parent, i) = uint32_t(i);
        return;
    }
    uint64_t start = At(prepareIn.row_offsets, i);
    uint64_t end   = At(prepareIn.row_offsets, i + 1);
    for (uint64_t e = start; e < end; ++e) {
        *Splice(prepareIn.edge_src, e) = uint32_t(i);
    }
}
