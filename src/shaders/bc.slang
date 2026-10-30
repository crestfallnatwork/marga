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

struct SharedBCFwdPayload {
    // Word offset of the (source, *) record row, first edge of the item, source
    // index and the item's sigma, plus the per-group edge prefix offsets.
    uint64_t row_base[cGroupItems];
    uint32_t edge_start[cGroupItems];
    uint32_t source[cGroupItems];
    float    sigma[cGroupItems];
    uint     offsets[cGroupItems + 1];
};
groupshared SharedBCFwdPayload bcFwdPayload;

struct BCFwdIn {
    uint64_t* row_offsets;
    uint32_t* column_indices;
    uint32_t* records_u;
    float*    records_f;
    uint64_t* frontier;
    uint64_t* next_frontier;
    uint32_t* counters;
    uint64_t num_vertices;
    uint64_t frontier_base;
    uint64_t frontier_count;
    uint64_t next_base;
    uint32_t level;
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

    uint running = 0;
    [unroll]
    for (uint i = 0; i < cItemsPerLane; ++i) {
        uint item = i * WAVE_SIZE + lid;
        uint64_t local = group_base + item;

        uint     source = 0;
        uint     start  = 0;
        uint     degree = 0;
        uint64_t row    = 0;
        float    sigma  = 0.0f;

        if (local < bcFwdIn.frontier_count) {
            uint64_t entry = At(bcFwdIn.frontier, bcFwdIn.frontier_base + local);
            source = uint(entry >> 32);
            uint u = uint(entry & 0xFFFFFFFFu);
            uint64_t first = At(bcFwdIn.row_offsets, u);
            start  = uint(first);
            degree = uint(At(bcFwdIn.row_offsets, uint64_t(u) + 1) - first);
            row    = uint64_t(source) * bcFwdIn.num_vertices * cRecordWords;
            sigma  = At(bcFwdIn.records_f, row + uint64_t(u) * cRecordWords + 1);
        }

        bcFwdPayload.row_base[item]   = row;
        bcFwdPayload.edge_start[item] = start;
        bcFwdPayload.source[item]     = source;
        bcFwdPayload.sigma[item]      = sigma;

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

    uint64_t eid = uint64_t(inPayload.edge_start[item]) + (edge - inPayload.offsets[item]);
    uint w       = At(bcFwdIn.column_indices, eid);
    uint64_t row = inPayload.row_base[item] + uint64_t(w) * cRecordWords;

    uint next_level = bcFwdIn.level + 1u;
    uint claimed    = 0;

    // Fast path: only the first edges that reach an unvisited vertex pay the
    // claim atomic.
    uint distance = At(bcFwdIn.records_u, row);
    if (distance == cUnvisited) {
        uint previous = 0;
        InterlockedCompareExchange(*Splice(bcFwdIn.records_u, row), cUnvisited, next_level, previous);
        if (previous == cUnvisited) {
            claimed   = 1;
            distance  = next_level;
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
    uint base        = 0;
    if (WaveIsFirstLane() && wave_claims > 0) {
        InterlockedAdd(*Splice(bcFwdIn.counters, 0), wave_claims, base);
    }
    base = WaveReadLaneFirst(base);

    if (claimed != 0) {
        *Splice(bcFwdIn.next_frontier, bcFwdIn.next_base + uint64_t(base) + uint64_t(wave_offset)) =
            (uint64_t(inPayload.source[item]) << 32) | uint64_t(w);
    }

    SetMeshOutputCounts(0, 0);
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
// @bind(bc_bwd)
[[vk::push_constant]]
uniform BCBwdIn bcBwdIn;

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
// the source itself at distance 0 with one path.
struct BCPrepareIn {
    uint32_t* records_u;
    float*    records_f;
    uint64_t  num_vertices;
    uint64_t  stride;
    uint64_t  num_sources;
};
// @bind(bc_prepare)
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

    uint64_t source = (uint64_t(tid.z) * bcPrepareIn.num_vertices) / bcPrepareIn.num_sources;
    if (v == source) {
        *Splice(bcPrepareIn.records_u, row)     = 0u;
        *Splice(bcPrepareIn.records_f, row + 1) = 1.0f;
    }
}
