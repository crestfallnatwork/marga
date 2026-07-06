// SPDX-License-Identifier: GPL-2.0-only
// Author: Himanshu Gupta <himanshu3@iisc.ac.in>

static const uint cMaxMeshDim = 65535;

struct SharedReductionTile {
    float32_t partial_errors[1024/WAVE_SIZE];
}
groupshared SharedReductionTile reductionTile;

struct ReductionIn {
    float32_t *past_pr;
    float32_t *current_pr;
    float32_t* d_delta;
    float32_t delta;
    float32_t d;
    uint64_t vertices_n;
    uint32_t* out_degrees;
    uint32_t is_restore_mode;
}
// @bind(reduce)
[[vk::push_constant]]
uniform ReductionIn reduceIn;
[shader("compute")]
[numthreads(1024, 1, 1)]
void reduce(uint32_t threadId : SV_DispatchThreadID,
            uint lid : SV_GroupIndex,
            uint gid : SV_GroupID,
) {
    float delta = 0.;
    if(threadId < reduceIn.vertices_n)  {
        if (reduceIn.is_restore_mode == 1) {
            uint32_t out_deg = reduceIn.out_degrees[threadId];
            if (out_deg > 0) {
                reduceIn.past_pr[threadId] *= out_deg;
            }
            return;
        }

        float dangling = reduceIn.current_pr[reduceIn.vertices_n]/reduceIn.vertices_n;
        float cur_pr = reduceIn.current_pr[threadId];
        cur_pr = (1 - reduceIn.d)/reduceIn.vertices_n +
                reduceIn.d*(cur_pr + dangling);

        uint32_t out_deg = reduceIn.out_degrees[threadId];
        float old_pr = reduceIn.past_pr[threadId];
        if (out_deg > 0) old_pr *= out_deg;

        delta = abs(old_pr - cur_pr);

        float pull_val = cur_pr;
        if (out_deg > 0) pull_val /= out_deg;

        reduceIn.current_pr[threadId] = pull_val;
        reduceIn.past_pr[threadId] = 0;

    }

    uint sgid = WaveGetWaveIndex();
    delta = WaveActiveSum(delta);
    if(WaveIsFirstLane()) {
       reductionTile.partial_errors[sgid] = delta;
    }

    GroupMemoryBarrierWithGroupSync();

    if(sgid == 0) {
        float32_t partial_error = 0;
        if(lid < 1024/WaveGetLaneCount()) {
            partial_error = reductionTile.partial_errors[lid];
        }
        delta = WaveActiveSum(partial_error);
        if(WaveIsFirstLane()) {
            InterlockedAdd(*reduceIn->d_delta, delta);
        }
    }
}

struct PageRankCSCIn{
    uint64_t* row_offsets;
    uint32_t* column_indices;
    float32_t *past_pr;
    float32_t *curr_pr;
    uint32_t *out_degrees;
    uint64_t vertices_n;
    uint64_t edges_n;
    uint32_t iter;
};

struct SharedPRCPayload {
    uint v_start;
    uint v_end;
};
groupshared SharedPRCPayload sharedPRPayload;

static const int PageRankVertexCount = 64;

// @bind(prc_task)
[[vk::push_constant]]
uniform PageRankCSCIn prIn;
[shader("amplification")]
[numthreads(1, 1, 1)]
void prc_task(
    uint threadId : SV_DispatchThreadID,
    uint lid : SV_GroupIndex,
) {

    uint32_t v_start = threadId * PageRankVertexCount;
    uint32_t v_end = v_start + PageRankVertexCount;
    if(v_end > prIn.vertices_n) v_end = prIn.vertices_n;

    uint32_t total_edges = prIn.row_offsets[v_end] - prIn.row_offsets[v_start];

    sharedPRPayload.v_start = v_start;
    sharedPRPayload.v_end = v_end;

    uint mesh_count = (total_edges + WAVE_SIZE - 1) / WAVE_SIZE;
    uint mesh_x = mesh_count > cMaxMeshDim ? cMaxMeshDim : mesh_count;
    uint mesh_y = mesh_count > cMaxMeshDim ?
                  ((mesh_count + cMaxMeshDim - 1) / cMaxMeshDim): 1;
    DispatchMesh(mesh_x, mesh_y, 1, sharedPRPayload);
}


[shader("mesh")]
[numthreads(WAVE_SIZE,1,1)]
[outputtopology("point")]
void prc_mesh(
    uint lid : SV_GroupIndex,
    uint3 GroupID : SV_GroupID,
    in payload SharedPRCPayload inPayload,
    out vertices float4 vertices[1],
    out indices uint indices[1]
) {
    uint gid_flat = GroupID.y * cMaxMeshDim + GroupID.x;
    uint local_offset = gid_flat * WAVE_SIZE + lid;

    uint32_t v_start = inPayload.v_start;
    uint32_t v_end = inPayload.v_end;

    uint32_t total_edges = prIn.row_offsets[v_end] - prIn.row_offsets[v_start];

    uint32_t dst = uint32_t.maxValue;
    float32_t pull = 0;
    uint dst_next = uint32_t.maxValue;
    if(local_offset < total_edges) {
        dst = v_start;
        uint64_t edge_offset = local_offset + prIn.row_offsets[dst];
        /* evil binary search */
        [unroll]
        for (uint step = (PageRankVertexCount >> 1); step > 0; step >>= 1) {
            if (dst + step < prIn.vertices_n && prIn.row_offsets[dst + step] <= edge_offset) {
                dst += step;
            }
        }

        uint src = *((uint32_t*)(((uint64_t)prIn.column_indices + edge_offset*4ull)));
        if (prIn.iter == 0) pull = prIn.past_pr[src]/prIn.out_degrees[src];
        else pull = prIn.past_pr[src];
    }

    for (uint offset = 1; offset < WAVE_SIZE; offset *= 2) {
        float32_t pull_left = WaveReadLaneAt(pull, lid-offset);
        uint dst_left = WaveReadLaneAt(dst, lid-offset);
        if (lid >= offset && dst_left == dst) {
            pull += pull_left;
        }
    }

    dst_next = WaveReadLaneAt(dst, lid+1);

    if(WAVE_SIZE - 1 == lid) {
        dst_next = uint32_t.maxValue;
    }

    bool write = (dst_next != dst);
    if(write) {
        InterlockedAdd(prIn.curr_pr[dst], pull);
    }

    SetMeshOutputCounts(0, 0);
}

struct DanglingIn{
    float32_t* past_pr;
    float32_t* curr_pr;
    uint* out_degree;
    uint64_t vertices_n;
}

struct DanglingSharedTile {
    float32_t partials[1024/WAVE_SIZE];
}
groupshared DanglingSharedTile danglingTile;
/* handle nodes with zero out degree*/
// @bind(dangling)
[[vk::push_constant]]
uniform DanglingIn danglingIn;
[shader("compute")]
[numthreads(1024,1,1)]
void dangling(uint32_t threadId : SV_DispatchThreadID,
            uint lid : SV_GroupIndex,
            uint gid : SV_GroupID,
) {
    float32_t dangling = 0;
    if(threadId < danglingIn.vertices_n) {
        uint outgoing = danglingIn.out_degree[threadId];
        if(outgoing == 0) {
            dangling = danglingIn.past_pr[threadId];
        }
    }
    uint sgid = WaveGetWaveIndex();
    dangling = WaveActiveSum(dangling);
    if(WaveIsFirstLane()) {
       danglingTile.partials[sgid] = dangling;
    }

    GroupMemoryBarrierWithGroupSync();

    if(sgid == 0) {
        float32_t partial = 0;
        if(lid < 1024/WaveGetLaneCount()) {
            partial = danglingTile.partials[lid];
        }
        partial = WaveActiveSum(partial);
        if(WaveIsFirstLane()) {
            InterlockedAdd(danglingIn.curr_pr[danglingIn.vertices_n], partial);
        }
    }
}


struct OutDegreeIn {
    uint64_t* row_offsets;
    uint32_t* column_indices;
    uint32_t* out_degrees;
    uint32_t vertices_n;
}

// @bind(outdegree)
[[vk::push_constant]]
uniform OutDegreeIn outDegreeIn;
[shader("compute")]
[numthreads(256,1,1)]
void outdegree(uint32_t threadId: SV_DispatchThreadID) {
    if(threadId >= outDegreeIn.vertices_n) {
        return;
    }
    uint32_t degree = outDegreeIn.row_offsets[threadId+1] - outDegreeIn.row_offsets[threadId];
    uint64_t start_col_addr = (uint64_t)outDegreeIn.column_indices + outDegreeIn.row_offsets[threadId]*4ull;
    for(uint32_t i = 0; i < degree; i++) {
        uint32_t src = *(uint32_t*)(start_col_addr + i*4ull);
        InterlockedAdd(outDegreeIn.out_degrees[src], 1);
    }
}

struct DispatchIn {
    float32_t* d_delta;
    float32_t* curr_pr;
    uint* iterations;
    uint* mesh_grid;
    uint* reduce_grid;
    uint vertices_n;
    float32_t delta;
}
// @bind(dispatch)
[[vk::push_constant]]
uniform DispatchIn dispatchIn;
[numthreads(1,1,1)]
[shader("compute")]
void dispatch() {
    uint* mesh_grid = dispatchIn.mesh_grid;
    uint* reduce_grid = dispatchIn.reduce_grid;
    if(*dispatchIn->d_delta > dispatchIn->delta) {
        *dispatchIn->d_delta = 0.;
        *dispatchIn->iterations = *(dispatchIn->iterations) + 1;
        dispatchIn.curr_pr[dispatchIn.vertices_n] = 0.;
        mesh_grid[0] = ((dispatchIn.vertices_n + PageRankVertexCount - 1)/PageRankVertexCount);
        reduce_grid[0] = ((dispatchIn.vertices_n + 1023)/1024);
    } else {
        mesh_grid[0] = 0;
        reduce_grid[0] = 0;
    }

    mesh_grid[1] = 1;
    reduce_grid[1] = 1;
    mesh_grid[2] = 1;
    reduce_grid[2] = 1;
}
