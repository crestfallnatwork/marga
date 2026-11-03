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
//     next phase seeds the cascade from the surviving vertices.
//  2. the work phase is one task dispatch. In cascade mode a task lane takes
//     one frontier vertex and sizes its row; the mesh stage tests each
//     out-edge, decrements the live neighbour and claims it once its degree
//     reaches k. In seed mode the task stage itself culls the survivor list
//     (removing a threshold's worth of vertices) and dispatches zero mesh
//     groups, so the culled work never reaches the rasterizer.
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
static const uint cStatePrevMode  = 12;  // mode of the phase that just ran (probe)
static const uint cStateMeshRuns  = 13;  // mesh-group counter at the last step (probe)

// Counters, u32.
static const uint cClaims       = 0;  // this phase's removed vertices
static const uint cKept         = 1;  // survivor list written by the last seed
static const uint cMinDeg       = 2;  // smallest degree kept by the last seed
static const uint cOverflow     = 3;  // frontier capacity violations
static const uint cSeeds        = 4;  // threshold seeds the sweep has started
static const uint cMeshRuns     = 5;  // mesh groups entered (probe only)
static const uint cSeedMeshRuns = 6;  // mesh groups entered in a seed phase (probe only)

T At<T>(T* ptr, uint64_t offset) {
    return *(T*)(uint64_t(ptr) + sizeof(T) * offset);
}
T* Splice<T>(T* ptr, uint64_t offset) {
    return (T*)(uint64_t(ptr) + sizeof(T) * offset);
}

struct SharedKCPayload {
    uint offsets[cGroupItems + 1];
    uint  k;
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
    // Debug: count mesh groups that actually run (and flag any that run in a
    // seed phase). Off in the benchmark path.
    uint32_t  probe;
};
// @bind(kc_task)
// @bind(kc_init_compute)
[[vk::push_constant]]
uniform KCIn kcIn;

// One work item per frontier vertex in cascade mode, or one surviving vertex
// per item in seed mode. The cascade payload is the per-item edge prefix sum;
// the mesh stage maps its edge id back to the item with a binary search over
// it. The seed is a pure cull: the task stage removes the threshold's vertices
// and returns a zero mesh dispatch.
[shader("amplification")]
[numthreads(WAVE_SIZE, 1, 1)]
void kc_task(
    uint3 gid : SV_GroupID,
    uint lid : SV_GroupIndex
) {
    uint g = (gid.y * cMaxMeshDim) + gid.x;
    uint64_t group_base = uint64_t(g) * cGroupItems;

    uint mode = uint(At(kcIn.state, cStateMode));
    uint k    = uint(At(kcIn.state, cStateK));
    uint count = (mode == cModeSeed) ? uint(At(kcIn.state, cStateSurvivors))
                                     : uint(At(kcIn.state, cStateCount));
    uint64_t base = At(kcIn.state, cStateBase);

    if (lid == 0) {
        kcPayload.k = k;
        // The mesh stage restarts its group ids with every DispatchMesh, so the
        // payload carries this task group's first frontier item.
        kcPayload.base = base + group_base;
    }

    if (mode == cModeSeed) {
        uint  src_idx = uint(At(kcIn.state, cStateSeedSrc));
        uint* src     = (src_idx != 0) ? kcIn.survivors1 : kcIn.survivors0;
        uint* dst     = (src_idx != 0) ? kcIn.survivors0 : kcIn.survivors1;

        uint64_t next_base = At(kcIn.state, cStateNext);
        uint64_t capacity  = At(kcIn.state, cStateCapacity);

        [unroll]
        for (uint i = 0; i < cItemsPerLane; ++i) {
            uint64_t local = group_base + i * WAVE_SIZE + lid;
            if (local < count) {
                uint v = At(src, local);

                uint word = v >> 5;
                uint mask = 1u << (v & 31);

                // A cascade of the previous threshold may already have claimed
                // it, and the survivor list is only compacted by a seed.
                if ((At(kcIn.removed, word) & mask) == 0) {
                    uint d = At(kcIn.deg, v);
                    if (d <= k) {
                        uint previous = 0;
                        InterlockedOr(*Splice(kcIn.removed, word), mask, previous);
                        if ((previous & mask) == 0) {
                            *Splice(kcIn.core, v) = k;
                            uint slot = 0;
                            InterlockedAdd(*Splice(kcIn.counters, cClaims), 1u, slot);
                            uint64_t pos = next_base + uint64_t(slot);
                            if (pos < capacity) {
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
                }
            }
        }

        DispatchMesh(0, 1, 1, kcPayload);
        return;
    }

    uint running = 0;
    [unroll]
    for (uint i = 0; i < cItemsPerLane; ++i) {
        uint item = i * WAVE_SIZE + lid;
        uint64_t local = group_base + item;

        uint degree = 0;
        if (local < count) {
            uint u     = At(kcIn.frontier, base + local);
            uint64_t s = At(kcIn.row_offsets, u);
            degree     = uint(At(kcIn.row_offsets, uint64_t(u) + 1) - s);
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

// Cascade only: one thread per out-edge, one test-and-claim per live target.
// Seed phases dispatch zero mesh groups, so this stage never sees a seed.
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
    if (kcIn.probe != 0 && lid == 0) {
        InterlockedAdd(*Splice(kcIn.counters, cMeshRuns), 1u);
    }

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
    // Debug: check that no mesh group ran during the phase that just finished
    // when that phase was a seed.
    uint32_t probe;
};
// @bind(kc_step_compute)
[[vk::push_constant]]
uniform KCStepIn kcStepIn;

[shader("compute")]
[numthreads(1, 1, 1)]
void kc_step_compute(uint3 tid : SV_DispatchThreadID) {
    uint32_t* grid = kcStepIn.grid;

    if (kcStepIn.probe != 0) {
        uint64_t runs = uint64_t(At(kcStepIn.counters, cMeshRuns));
        uint64_t prev = At(kcStepIn.state, cStateMeshRuns);
        if (At(kcStepIn.state, cStatePrevMode) == cModeSeed) {
            // Mesh groups that ran during the seed phase just completed. A seed
            // dispatches zero, so this must stay at zero.
            *Splice(kcStepIn.counters, cSeedMeshRuns) =
                At(kcStepIn.counters, cSeedMeshRuns) + uint(runs - prev);
        }
        *Splice(kcStepIn.state, cStateMeshRuns) = runs;
    }

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

    // Remember which phase this step just launched so the probe can attribute
    // the next interval's mesh groups to it.
    *Splice(kcStepIn.state, cStatePrevMode) = At(kcStepIn.state, cStateMode);
}
