// SPDX-License-Identifier: GPL-2.0-only
// Author: Himanshu Gupta <himanshu3@iisc.ac.in>

// Multi-source Brandes betweenness centrality.
//
// Per (source, vertex) record: {distance, path count sigma, dependency delta,
// pad} in one 16 B entry, so a vertex touches one 32 B sector per access.
// Sources are strided (source i = i * V / k) and share one frontier, packed as
// (source << 32 | vertex) u64 entries.
//
// Forward: level-synchronous path counting. The task stage sizes the work from
// the frontier items' out-degrees, stages the per-item row base, first edge and
// sigma in the payload, and the mesh stage performs one test-and-claim plus one
// sigma accumulation per out-edge. Claims append to the next frontier with one
// reservation per wave.
//
// Backward: reverse level order, one thread per (source, vertex) item that
// gathers over its out-edges and accumulates delta in a register, so no atomics
// and no cross-item races are involved.
//
// Records are accessed through two typed views of the same buffer (u32 for the
// distance word and the atomic claim, f32 for sigma and delta); the word layout
// is cRecordWords words per entry.

static const uint cMaxMeshDim   = 65535;
static const uint cItemsPerLane = 8;
static const uint cGroupItems   = WAVE_SIZE * cItemsPerLane;
static const uint cRecordWords  = 4;
static const uint cMeshGroup    = 128;
static const uint cUnvisited    = 0xFFFFFFFFu;

T At<T>(T* ptr, uint64_t offset) {
    return *(T*)(uint64_t(ptr) + sizeof(T) * offset);
}
T* Splice<T>(T* ptr, uint64_t offset) {
    return (T*)(uint64_t(ptr) + sizeof(T) * offset);
}

// The payload is copied once per mesh workgroup, so it holds only what the
// mesh stage cannot re-derive cheaply: the item's sigma (a random read of its
// own record) and the edge prefix offsets. The frontier entry and the item's
// first edge are re-read from global, where a mesh group's 128 consecutive
// edges hit one or two sectors of each.
struct SharedBCFwdPayload {
    // Frontier index of this group's first item; the mesh stage's own group id
    // restarts with every DispatchMesh.
    uint64_t base;
    float    sigma[cGroupItems];
    uint     offsets[cGroupItems + 1];
};
groupshared SharedBCFwdPayload bcFwdPayload;

// Level state on the device, so a whole batch of levels runs from one host
// submission: [0] frontier base, [1] frontier count, [2] next base,
// [3] levels recorded, [4] frontier capacity. counters[0] holds the claims of
// the running level and is reset by bc_step_compute.
struct BCFwdIn {
    uint64_t* row_offsets;
    uint32_t* column_indices;
    uint32_t* records_u;
    float*    records_f;
    uint64_t* frontier;
    uint64_t* level_state;
    uint32_t* counters;
    uint64_t  num_vertices;
    uint32_t  level;
};
// @bind(bc_fwd)
[[vk::push_constant]]
uniform BCFwdIn bcFwdIn;

[shader("amplification")]
[numthreads(WAVE_SIZE, 1, 1)]
void bc_fwd_task(
    uint3 gid : SV_GroupID,
    uint lid : SV_GroupIndex
) {
    uint g = (gid.y * cMaxMeshDim) + gid.x;
    uint64_t group_base = uint64_t(g) * cGroupItems;

    uint64_t base  = At(bcFwdIn.level_state, 0);
    uint64_t count = At(bcFwdIn.level_state, 1);

    if (lid == 0) {
        bcFwdPayload.base = base + group_base;
    }

    uint running = 0;
    [unroll]
    for (uint i = 0; i < cItemsPerLane; ++i) {
        uint item = i * WAVE_SIZE + lid;
        uint64_t local = group_base + item;

        uint  degree = 0;
        float sigma  = 0.0f;

        if (local < count) {
            uint64_t entry = At(bcFwdIn.frontier, base + local);
            uint     source = uint(entry >> 32);
            uint     u      = uint(entry & 0xFFFFFFFFu);
            uint64_t first  = At(bcFwdIn.row_offsets, u);
            degree = uint(At(bcFwdIn.row_offsets, uint64_t(u) + 1) - first);
            uint64_t row = uint64_t(source) * bcFwdIn.num_vertices * cRecordWords;
            sigma = At(bcFwdIn.records_f, row + uint64_t(u) * cRecordWords + 1);
        }

        bcFwdPayload.sigma[item] = sigma;

        uint prefix = WavePrefixSum(degree);
        uint total  = WaveActiveSum(degree);
        bcFwdPayload.offsets[item] = running + prefix;
        running += total;
    }

    if (lid == WAVE_SIZE - 1) {
        bcFwdPayload.offsets[cGroupItems] = running;
    }

    GroupMemoryBarrierWithGroupSync();

    uint mesh_count = (running + cMeshGroup - 1) / cMeshGroup;
    uint mesh_x     = min(mesh_count, cMaxMeshDim);
    uint mesh_y     = mesh_count > cMaxMeshDim ? ((mesh_count + cMaxMeshDim - 1) / cMaxMeshDim) : 1;

    DispatchMesh(mesh_x, mesh_y, 1, bcFwdPayload);
}

[shader("mesh")]
[numthreads(128, 1, 1)]
[outputtopology("point")]
void bc_fwd_mesh(
    uint lid : SV_GroupIndex,
    uint3 gid : SV_GroupID,
    in payload SharedBCFwdPayload inPayload,
    out vertices float4 vertices[1],
    out indices uint indices[1]
) {
    uint g      = (gid.y * cMaxMeshDim) + gid.x;
    uint edge   = g * cMeshGroup + lid;
    uint total  = inPayload.offsets[cGroupItems];
    if (edge >= total) {
        SetMeshOutputCounts(0, 0);
        return;
    }

    // Map the mesh thread to its frontier item within the task group.
    uint item = 0;
    [unroll]
    for (uint step = (cGroupItems >> 1); step > 0; step >>= 1) {
        if (inPayload.offsets[item + step] <= edge) {
            item += step;
        }
    }

    // Re-derive the item's vertex, row and first edge from global memory; the
    // group's 128 consecutive edges touch one or two sectors of each.
    uint64_t entry  = At(bcFwdIn.frontier, inPayload.base + uint64_t(item));
    uint     source = uint(entry >> 32);
    uint     u      = uint(entry & 0xFFFFFFFFu);
    uint64_t row    = uint64_t(source) * bcFwdIn.num_vertices * cRecordWords;
    uint64_t eid    = At(bcFwdIn.row_offsets, u) + (edge - inPayload.offsets[item]);
    uint     w      = At(bcFwdIn.column_indices, eid);
    row += uint64_t(w) * cRecordWords;

    uint next_level = bcFwdIn.level + 1u;
    uint claimed    = 0;

    // Fast path: only edges that reach a still-unvisited vertex pay the claim
    // atomic.
    uint distance = At(bcFwdIn.records_u, row);
    if (distance == cUnvisited) {
        uint previous = 0;
        InterlockedCompareExchange(*Splice(bcFwdIn.records_u, row), cUnvisited, next_level, previous);
        if (previous == cUnvisited) {
            claimed  = 1;
            distance = next_level;
        } else {
            distance = previous;
        }
    }

    // Every edge landing on the next level contributes paths, not only the
    // claiming one; sigma of the source vertex is final after the previous
    // level's barrier, so float accumulation order does not matter.
    if (distance == next_level) {
        InterlockedAdd(*Splice(bcFwdIn.records_f, row + 1), inPayload.sigma[item]);
    }

    uint wave_claims = WaveActiveCountBits(claimed != 0u);
    uint wave_offset = WavePrefixCountBits(claimed != 0u);
    uint reserved    = 0;
    if (WaveIsFirstLane() && wave_claims > 0) {
        InterlockedAdd(*Splice(bcFwdIn.counters, 0), wave_claims, reserved);
    }
    reserved = WaveReadLaneFirst(reserved);

    if (claimed != 0) {
        uint64_t slot = At(bcFwdIn.level_state, 2) + uint64_t(reserved) + uint64_t(wave_offset);
        if (slot < At(bcFwdIn.level_state, 4)) {
            *Splice(bcFwdIn.frontier, slot) = (uint64_t(source) << 32) | uint64_t(w);
        }
    }

    SetMeshOutputCounts(0, 0);
}

// One thread that advances the device-side level state by one level: records
// the level it is about to run, consumes the previous level's claim count,
// resets the claim counter and writes the indirect task grid for the level.
struct BCStepIn {
    uint64_t* level_state;
    uint64_t* levels;
    uint32_t* counters;
    uint32_t* grid;
    uint64_t  levels_capacity;
};
// @bind(bc_step_compute)
[[vk::push_constant]]
uniform BCStepIn bcStepIn;

[shader("compute")]
[numthreads(1, 1, 1)]
void bc_step_compute(uint3 tid : SV_DispatchThreadID) {
    // The pending claim count of the previous level is this level's item count
    // (every claim is a distinct vertex), and state[2] already holds this
    // level's frontier base.
    uint64_t base     = At(bcStepIn.level_state, 2);
    uint64_t produced = uint64_t(At(bcStepIn.counters, 0));
    *Splice(bcStepIn.level_state, 0) = base;
    *Splice(bcStepIn.level_state, 1) = produced;
    *Splice(bcStepIn.level_state, 2) = base + produced;
    *Splice(bcStepIn.counters, 0)    = 0u;

    // An empty level means the traversal is over, so only non-empty levels are
    // recorded and the recorded index stays equal to the level number.
    uint64_t recorded = At(bcStepIn.level_state, 3);
    if (produced > 0 && recorded < bcStepIn.levels_capacity) {
        *Splice(bcStepIn.levels, 2 * recorded)     = base;
        *Splice(bcStepIn.levels, 2 * recorded + 1) = produced;
        *Splice(bcStepIn.level_state, 3)           = recorded + 1;
    }

    uint64_t groups = (produced + cGroupItems - 1) / cGroupItems;
    uint64_t gx     = min(groups, uint64_t(cMaxMeshDim));
    uint64_t gy     = groups > cMaxMeshDim ? ((groups + cMaxMeshDim - 1) / cMaxMeshDim) : 1;
    *Splice(bcStepIn.grid, 0) = uint32_t(gx);
    *Splice(bcStepIn.grid, 1) = uint32_t(gy);
    *Splice(bcStepIn.grid, 2) = 1u;
}

struct BCBwdIn {
    uint64_t* row_offsets;
    uint32_t* column_indices;
    uint32_t* records_u;
    float*    records_f;
    uint64_t* frontier;
    uint64_t frontier_base;
    uint64_t frontier_count;
    uint64_t num_vertices;
    uint64_t stride;
    uint32_t level;
};
// @bind(bc_bwd_compute)
// @bind(bc_bwdm_task)
[[vk::push_constant]]
uniform BCBwdIn bcBwdIn;

// Scatter form of the dependency pass: the task stage stages the target delta
// slot of every item, the mesh stage adds one contribution per out-edge that
// lands on the next level. Used for the gather/scatter comparison.
struct SharedBCBwdPayload {
    uint64_t delta_word[cGroupItems];
    uint64_t row_base[cGroupItems];
    uint32_t edge_start[cGroupItems];
    float    sigma[cGroupItems];
    uint     offsets[cGroupItems + 1];
};
groupshared SharedBCBwdPayload bcBwdPayload;

[shader("amplification")]
[numthreads(WAVE_SIZE, 1, 1)]
void bc_bwdm_task(
    uint3 gid : SV_GroupID,
    uint lid : SV_GroupIndex
) {
    uint g = (gid.y * cMaxMeshDim) + gid.x;
    uint64_t group_base = uint64_t(g) * cGroupItems;

    uint running = 0;
    [unroll]
    for (uint i = 0; i < cItemsPerLane; ++i) {
        uint item = i * WAVE_SIZE + lid;
        uint64_t local = group_base + item;

        uint     start  = 0;
        uint     degree = 0;
        uint64_t row    = 0;
        uint64_t delta  = 0;
        float    sigma  = 0.0f;

        if (local < bcBwdIn.frontier_count) {
            uint64_t entry = At(bcBwdIn.frontier, bcBwdIn.frontier_base + local);
            uint     source = uint(entry >> 32);
            uint     u      = uint(entry & 0xFFFFFFFFu);
            uint64_t first  = At(bcBwdIn.row_offsets, u);
            start  = uint(first);
            degree = uint(At(bcBwdIn.row_offsets, uint64_t(u) + 1) - first);
            row    = uint64_t(source) * bcBwdIn.num_vertices * cRecordWords;
            delta  = row + uint64_t(u) * cRecordWords + 2;
            sigma  = At(bcBwdIn.records_f, row + uint64_t(u) * cRecordWords + 1);
        }

        bcBwdPayload.delta_word[item] = delta;
        bcBwdPayload.row_base[item]   = row;
        bcBwdPayload.edge_start[item] = start;
        bcBwdPayload.sigma[item]      = sigma;

        uint prefix = WavePrefixSum(degree);
        uint total  = WaveActiveSum(degree);
        bcBwdPayload.offsets[item] = running + prefix;
        running += total;
    }

    if (lid == WAVE_SIZE - 1) {
        bcBwdPayload.offsets[cGroupItems] = running;
    }

    GroupMemoryBarrierWithGroupSync();

    uint mesh_count = (running + cMeshGroup - 1) / cMeshGroup;
    uint mesh_x     = min(mesh_count, cMaxMeshDim);
    uint mesh_y     = mesh_count > cMaxMeshDim ? ((mesh_count + cMaxMeshDim - 1) / cMaxMeshDim) : 1;

    DispatchMesh(mesh_x, mesh_y, 1, bcBwdPayload);
}

[shader("mesh")]
[numthreads(128, 1, 1)]
[outputtopology("point")]
void bc_bwdm_mesh(
    uint lid : SV_GroupIndex,
    uint3 gid : SV_GroupID,
    in payload SharedBCBwdPayload inPayload,
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

    uint item = 0;
    [unroll]
    for (uint step = (cGroupItems >> 1); step > 0; step >>= 1) {
        if (inPayload.offsets[item + step] <= edge) {
            item += step;
        }
    }

    uint64_t eid  = uint64_t(inPayload.edge_start[item]) + (edge - inPayload.offsets[item]);
    uint     w    = At(bcBwdIn.column_indices, eid);
    uint64_t slot = inPayload.row_base[item] + uint64_t(w) * cRecordWords;

    // Delta of the successor is final: it sits one level deeper and was
    // processed by the previous dispatch of the reverse sweep.
    if (At(bcBwdIn.records_u, slot) == bcBwdIn.level + 1u) {
        float contribution = inPayload.sigma[item] / At(bcBwdIn.records_f, slot + 1) *
                             (1.0f + At(bcBwdIn.records_f, slot + 2));
        InterlockedAdd(*Splice(bcBwdIn.records_f, inPayload.delta_word[item]), contribution);
    }

    SetMeshOutputCounts(0, 0);
}

[shader("compute")]
[numthreads(128, 1, 1)]
void bc_bwd_compute(uint3 tid : SV_DispatchThreadID) {
    uint64_t item = uint64_t(tid.y) * bcBwdIn.stride + tid.x;
    if (item >= bcBwdIn.frontier_count) {
        return;
    }

    uint64_t entry  = At(bcBwdIn.frontier, bcBwdIn.frontier_base + item);
    uint     source = uint(entry >> 32);
    uint     u      = uint(entry & 0xFFFFFFFFu);
    uint64_t row    = uint64_t(source) * bcBwdIn.num_vertices * cRecordWords;
    uint64_t first  = At(bcBwdIn.row_offsets, u);

    float sigma_u = At(bcBwdIn.records_f, row + uint64_t(u) * cRecordWords + 1);
    float acc     = 0.0f;

    uint64_t end = At(bcBwdIn.row_offsets, uint64_t(u) + 1);
    for (uint64_t e = first; e < end; ++e) {
        uint     w    = At(bcBwdIn.column_indices, e);
        uint64_t slot = row + uint64_t(w) * cRecordWords;
        if (At(bcBwdIn.records_u, slot) == bcBwdIn.level + 1u) {
            acc += sigma_u / At(bcBwdIn.records_f, slot + 1) *
                   (1.0f + At(bcBwdIn.records_f, slot + 2));
        }
    }

    *Splice(bcBwdIn.records_f, row + uint64_t(u) * cRecordWords + 2) = acc;
}

// One-time setup per source row: distance unvisited, sigma and delta zero, and
// the source itself at distance 0 with one path. Row z belongs to source
// sources[z] (the sample minus its sources without out-edges).
struct BCPrepareIn {
    uint32_t* records_u;
    float*    records_f;
    uint32_t* sources;
    uint64_t  num_vertices;
    uint64_t  stride;
};
// @bind(bc_prepare_compute)
[[vk::push_constant]]
uniform BCPrepareIn bcPrepareIn;

[shader("compute")]
[numthreads(256, 1, 1)]
void bc_prepare_compute(uint3 tid : SV_DispatchThreadID) {
    uint64_t v = uint64_t(tid.y) * bcPrepareIn.stride + tid.x;
    if (v >= bcPrepareIn.num_vertices) {
        return;
    }

    uint64_t row = (uint64_t(tid.z) * bcPrepareIn.num_vertices + v) * cRecordWords;
    *Splice(bcPrepareIn.records_u, row)     = cUnvisited;
    *Splice(bcPrepareIn.records_f, row + 1) = 0.0f;
    *Splice(bcPrepareIn.records_f, row + 2) = 0.0f;

    uint64_t source = uint64_t(At(bcPrepareIn.sources, tid.z));
    if (v == source) {
        *Splice(bcPrepareIn.records_u, row)     = 0u;
        *Splice(bcPrepareIn.records_f, row + 1) = 1.0f;
    }
}
