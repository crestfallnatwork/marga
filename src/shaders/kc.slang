// SPDX-License-Identifier: GPL-2.0-only
// Author: Himanshu Gupta <himanshu3@iisc.ac.in>

// Core-number decomposition over a symmetric simple graph (self-loops
// excluded). The sweep raises k from the smallest degree present; at each
// threshold every remaining vertex whose degree has dropped to k or less is
// removed with core[v] = k, and its live neighbours lose one from their
// remaining degree.
//
// A device-side phase chain drives the sweep (as bc's level chain does):
//  1. kc_step_compute consumes the previous phase's claims. If there are any,
//     the next phase is a cascade wave over them; otherwise k advances and the
//     next phase partitions the surviving vertices, seeding the cascade.
//  2. the work phase is one task/mesh dispatch. In cascade mode a task lane
//     takes one frontier vertex and sizes its row; the mesh stage tests each
//     out-edge, decrements the live neighbour and claims it once its degree
//     reaches k. In seed mode an item is one surviving vertex, repartitioned
//     into the next survivor list or removed into the frontier.
// The host submits batches of phases and only reads the state between batches.

static const uint cMaxMeshDim   = 65535;
static const uint cItemsPerLane = 4;
static const uint cGroupItems   = WAVE_SIZE * cItemsPerLane;
static const uint cMeshGroup    = 128;
static const uint cUnset        = 0xFFFFFFFFu;
static const uint cMinusOne     = 0xFFFFFFFFu;

// Phase of the work dispatch.
static const uint cModeCascade = 0;
static const uint cModeSeed    = 1;

// Device state, u64 slots. Written by kc_step_compute and the host.
static const uint cStateK         = 0;  // current threshold
static const uint cStateBase      = 1;  // first frontier entry of this wave
static const uint cStateCount     = 2;  // cascade frontier length
static const uint cStateNext      = 3;  // where this phase appends its claims
static const uint cStateSeedRead  = 4;  // survivor list the next seed reads
static const uint cStateDone      = 5;
static const uint cStateMode      = 6;
static const uint cStateSurvivors = 7;  // survivor list length this seed scans
static const uint cStateCapacity  = 8;  // frontier capacity in entries
static const uint cStateSeedSrc   = 9;  // survivor list this seed reads
static const uint cStatePhases    = 10;
static const uint cStatePeak      = 11;

// Counters, u32.
static const uint cClaims   = 0;  // this phase's removed vertices
static const uint cKept     = 1;  // survivor list written by the last seed
static const uint cMinDeg   = 2;  // smallest degree kept by the last seed
static const uint cOverflow = 3;  // frontier capacity violations
static const uint cSeeds    = 4;  // threshold seeds the sweep has started

T At<T>(T* ptr, uint64_t offset) {
    return *(T*)(uint64_t(ptr) + sizeof(T) * offset);
}
T* Splice<T>(T* ptr, uint64_t offset) {
    return (T*)(uint64_t(ptr) + sizeof(T) * offset);
}

struct SharedKCPayload {
    uint offsets[cGroupItems + 1];
    // Copy of the step's phase choice, so the mesh stage does not re-read it.
    uint  mode;
    uint  k;
    uint  seed_src;
    uint64_t base;
};
groupshared SharedKCPayload kcPayload;

struct KCIn {
    uint64_t* row_offsets;
    uint32_t* column_indices;
    uint64_t* state;
    uint32_t* counters;
    uint32_t* deg;
    uint32_t* core;
    uint32_t* removed;
    uint32_t* survivors0;
    uint32_t* survivors1;
    uint32_t* frontier;
    uint64_t  num_vertices;
    uint64_t  stride;
};
// @bind(kc_task)
// @bind(kc_init_compute)
[[vk::push_constant]]
uniform KCIn kcIn;

// One work item per frontier vertex (cascade) or per surviving vertex (seed).
// The payload is the per-item edge prefix sum; the mesh stage maps its edge id
// back to the item with a binary search over it.
[shader("amplification")]
[numthreads(WAVE_SIZE, 1, 1)]
void kc_task(
    uint3 gid : SV_GroupID,
    uint lid : SV_GroupIndex
) {
    uint g = (gid.y * cMaxMeshDim) + gid.x;
    uint64_t group_base = uint64_t(g) * cGroupItems;

    uint mode = uint(At(kcIn.state, cStateMode));
    uint count = (mode == cModeSeed) ? uint(At(kcIn.state, cStateSurvivors))
                                     : uint(At(kcIn.state, cStateCount));
    uint64_t base = At(kcIn.state, cStateBase);

    if (lid == 0) {
        kcPayload.mode     = mode;
        kcPayload.k        = uint(At(kcIn.state, cStateK));
        kcPayload.seed_src = uint(At(kcIn.state, cStateSeedSrc));
        // The mesh stage restarts its group ids with every DispatchMesh, so the
        // payload carries this task group's first item. A seed scans the whole
        // survivor list from the front.
        kcPayload.base     = (mode == cModeSeed) ? group_base : (base + group_base);
    }

    uint running = 0;
    [unroll]
    for (uint i = 0; i < cItemsPerLane; ++i) {
        uint item = i * WAVE_SIZE + lid;
        uint64_t local = group_base + item;

        uint degree = 0;
        if (local < count) {
            degree = 1;
            if (mode == cModeCascade) {
                uint u      = At(kcIn.frontier, base + local);
                uint64_t s  = At(kcIn.row_offsets, u);
                degree      = uint(At(kcIn.row_offsets, uint64_t(u) + 1) - s);
            }
        }

        uint prefix = WavePrefixSum(degree);
        uint total  = WaveActiveSum(degree);
        kcPayload.offsets[item] = running + prefix;
        running += total;
    }

    if (lid == WAVE_SIZE - 1) {
        kcPayload.offsets[cGroupItems] = running;
    }

    GroupMemoryBarrierWithGroupSync();

    uint mesh_count = (running + cMeshGroup - 1) / cMeshGroup;
    uint mesh_x     = min(mesh_count, cMaxMeshDim);
    uint mesh_y     = mesh_count > cMaxMeshDim ? ((mesh_count + cMaxMeshDim - 1) / cMaxMeshDim) : 1;

    DispatchMesh(mesh_x, mesh_y, 1, kcPayload);
}

[shader("mesh")]
[numthreads(128, 1, 1)]
[outputtopology("point")]
void kc_mesh(
    uint lid : SV_GroupIndex,
    uint3 gid : SV_GroupID,
    in payload SharedKCPayload inPayload,
    out vertices float4 vertices[1],
    out indices uint indices[1]
) {
    uint g     = (gid.y * cMaxMeshDim) + gid.x;
    uint edge  = g * cMeshGroup + lid;
    uint total = inPayload.offsets[cGroupItems];
    if (edge >= total) {
        SetMeshOutputCounts(0, 0);
        return;
    }

    // Map the mesh thread to its item within the task group.
    uint item = 0;
    [unroll]
    for (uint step = (cGroupItems >> 1); step > 0; step >>= 1) {
        if (inPayload.offsets[item + step] <= edge) {
            item += step;
        }
    }

    uint k = inPayload.k;

    if (inPayload.mode == cModeSeed) {
        uint* src = (inPayload.seed_src != 0) ? kcIn.survivors1 : kcIn.survivors0;
        uint* dst = (inPayload.seed_src != 0) ? kcIn.survivors0 : kcIn.survivors1;
        uint  v   = At(src, inPayload.base + uint64_t(item));

        uint word = v >> 5;
        uint mask = 1u << (v & 31);

        // A cascade of the previous threshold may already have claimed it.
        if ((At(kcIn.removed, word) & mask) != 0) {
            SetMeshOutputCounts(0, 0);
            return;
        }

        uint d = At(kcIn.deg, v);
        if (d <= k) {
            uint previous = 0;
            InterlockedOr(*Splice(kcIn.removed, word), mask, previous);
            if ((previous & mask) == 0) {
                *Splice(kcIn.core, v) = k;
                uint slot = 0;
                InterlockedAdd(*Splice(kcIn.counters, cClaims), 1u, slot);
                uint64_t pos = At(kcIn.state, cStateNext) + uint64_t(slot);
                if (pos < At(kcIn.state, cStateCapacity)) {
                    *Splice(kcIn.frontier, pos) = v;
                } else {
                    InterlockedAdd(*Splice(kcIn.counters, cOverflow), 1u);
                }
            }
        } else {
            uint slot = 0;
            InterlockedAdd(*Splice(kcIn.counters, cKept), 1u, slot);
            *Splice(dst, slot) = v;
            InterlockedMin(*Splice(kcIn.counters, cMinDeg), d);
        }
    } else {
        uint     u   = At(kcIn.frontier, inPayload.base + uint64_t(item));
        uint64_t eid = At(kcIn.row_offsets, u) + uint64_t(edge - inPayload.offsets[item]);
        uint     w   = At(kcIn.column_indices, eid);

        // Self-loops are removed by the importer; skip any that remain.
        if (w == u) {
            SetMeshOutputCounts(0, 0);
            return;
        }

        uint word = w >> 5;
        uint mask = 1u << (w & 31);
        if ((At(kcIn.removed, word) & mask) != 0) {
            SetMeshOutputCounts(0, 0);
            return;
        }

        // The returned value is the degree before this decrement. Racing
        // decrements can only push it further down; the claim below is still
        // taken exactly once.
        uint old = 0;
        InterlockedAdd(*Splice(kcIn.deg, w), cMinusOne, old);
        uint next = old - 1u;

        if (next <= k) {
            uint previous = 0;
            InterlockedOr(*Splice(kcIn.removed, word), mask, previous);
            if ((previous & mask) == 0) {
                *Splice(kcIn.core, w) = k;
                uint slot = 0;
                InterlockedAdd(*Splice(kcIn.counters, cClaims), 1u, slot);
                uint64_t pos = At(kcIn.state, cStateNext) + uint64_t(slot);
                if (pos < At(kcIn.state, cStateCapacity)) {
                    *Splice(kcIn.frontier, pos) = w;
                } else {
                    InterlockedAdd(*Splice(kcIn.counters, cOverflow), 1u);
                }
            }
        }
    }

    SetMeshOutputCounts(0, 0);
}

// Initial state: remaining degree from the row length, every core unset, and
// degree-0 vertices removed with core 0. The non-isolated vertices seed the
// first survivor list.
[shader("compute")]
[numthreads(256, 1, 1)]
void kc_init_compute(uint3 tid : SV_DispatchThreadID) {
    uint64_t v = uint64_t(tid.y) * kcIn.stride + tid.x;
    if (v >= kcIn.num_vertices) {
        return;
    }

    uint64_t first = At(kcIn.row_offsets, v);
    uint     d     = uint(At(kcIn.row_offsets, v + 1) - first);

    *Splice(kcIn.deg, v)  = d;
    *Splice(kcIn.core, v) = cUnset;

    if (d == 0) {
        *Splice(kcIn.core, v) = 0u;
        uint word = uint(v) >> 5;
        uint mask = 1u << (uint(v) & 31);
        uint previous = 0;
        InterlockedOr(*Splice(kcIn.removed, word), mask, previous);
    } else {
        uint slot = 0;
        InterlockedAdd(*Splice(kcIn.counters, cKept), 1u, slot);
        *Splice(kcIn.survivors0, slot) = uint(v);
        InterlockedMin(*Splice(kcIn.counters, cMinDeg), d);
    }
}

// Advances the phase chain by one work dispatch: consume the previous phase's
// claims, or start the next threshold, and write the work dispatch's grid.
struct KCStepIn {
    uint64_t* state;
    uint32_t* counters;
    uint32_t* grid;
};
// @bind(kc_step_compute)
[[vk::push_constant]]
uniform KCStepIn kcStepIn;

[shader("compute")]
[numthreads(1, 1, 1)]
void kc_step_compute(uint3 tid : SV_DispatchThreadID) {
    uint32_t* grid = kcStepIn.grid;

    if (At(kcStepIn.state, cStateDone) != 0) {
        grid[0] = 0;
        grid[1] = 1;
        grid[2] = 1;
        return;
    }

    *Splice(kcStepIn.state, cStatePhases) = At(kcStepIn.state, cStatePhases) + 1;

    uint64_t pending = uint64_t(At(kcStepIn.counters, cClaims));
    *Splice(kcStepIn.counters, cClaims) = 0u;

    uint64_t items = 0;
    if (pending > 0) {
        uint64_t base = At(kcStepIn.state, cStateNext);
        uint64_t next = base + pending;

        *Splice(kcStepIn.state, cStateBase)  = base;
        *Splice(kcStepIn.state, cStateCount) = pending;
        *Splice(kcStepIn.state, cStateNext)  = next;
        *Splice(kcStepIn.state, cStateMode)  = cModeCascade;

        if (next > At(kcStepIn.state, cStatePeak)) {
            *Splice(kcStepIn.state, cStatePeak) = next;
        }
        if (next > At(kcStepIn.state, cStateCapacity)) {
            InterlockedAdd(*Splice(kcStepIn.counters, cOverflow), 1u);
        }
        items = pending;
    } else {
        // Every vertex is either removed or still has degree > k, so the last
        // seed's kept count is the surviving set the next threshold scans.
        uint64_t alive = uint64_t(At(kcStepIn.counters, cKept));
        if (alive == 0) {
            *Splice(kcStepIn.state, cStateDone) = 1;
            grid[0] = 0;
            grid[1] = 1;
            grid[2] = 1;
            return;
        }

        uint64_t k       = At(kcStepIn.state, cStateK);
        uint64_t min_deg = uint64_t(At(kcStepIn.counters, cMinDeg));
        uint64_t src     = At(kcStepIn.state, cStateSeedRead);

        // A seed that claimed nothing scanned the current degrees, so its
        // minimum is the next threshold that can remove anything. After a
        // cascade the stored minimum is stale (the cascade lowered degrees), so
        // only raise k by one and let the next seed recompute it.
        uint64_t next_k = k + 1;
        if (At(kcStepIn.state, cStateMode) == cModeSeed) {
            next_k = max(next_k, min_deg);
        }

        *Splice(kcStepIn.state, cStateK)         = next_k;
        *Splice(kcStepIn.state, cStateSurvivors) = alive;
        *Splice(kcStepIn.state, cStateSeedSrc)   = src;
        *Splice(kcStepIn.state, cStateSeedRead)  = 1 - src;
        *Splice(kcStepIn.state, cStateMode)      = cModeSeed;

        *Splice(kcStepIn.counters, cKept)   = 0u;
        *Splice(kcStepIn.counters, cMinDeg) = cUnset;
        *Splice(kcStepIn.counters, cSeeds)  = At(kcStepIn.counters, cSeeds) + 1u;
        items = alive;
    }

    uint64_t groups = (items + cGroupItems - 1) / cGroupItems;
    uint64_t gx     = min(groups, uint64_t(cMaxMeshDim));
    uint64_t gy     = groups > cMaxMeshDim ? ((groups + cMaxMeshDim - 1) / cMaxMeshDim) : 1;
    grid[0] = uint(gx);
    grid[1] = uint(gy);
    grid[2] = 1;
}
