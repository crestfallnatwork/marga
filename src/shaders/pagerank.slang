struct ReductionFlag{
    uint32_t padding[31];
    uint32_t flag;
}
[[vk::push_constant]]
uniform ReductionFlag _dummy_reduction_flag; /*dummy push constant for reflection support*/

struct ReductionIn {
    float32_t *past_pr;
    float32_t *current_pr;
    ReductionFlag *flag;
    float32_t delta;
    uint64_t vertices_n;
    float32_t d;
}
// @bind(reduce)
[[vk::push_constant]]
uniform ReductionIn reduceIn;
[shader("compute")]
[numthreads(256, 1, 1)]
void reduce(uint32_t threadId : SV_DispatchThreadID) {
    if(threadId >= reduceIn.vertices_n) return;

    float cur_pr = reduceIn.current_pr[threadId];
    cur_pr = (1 - reduceIn.d)/reduceIn.vertices_n +
              reduceIn.d*cur_pr;

    float32_t delta = abs(reduceIn.past_pr[threadId] - cur_pr);

    reduceIn.current_pr[threadId] = cur_pr;
    reduceIn.past_pr[threadId] = 0;

    if(delta > reduceIn.delta && (reduceIn.flag->flag == false)) {
        /* Yes indeed! This is a race but the gods are good.
         * This is the good type, the fun type. The type
         * that only hurts atomic contention and nothing else
         */
        reduceIn.flag->flag = true;
    }
}

struct DispatchIn {
    uint64_t edges_n;
    ReductionFlag *flag;
    uint *grid;
}
// @bind(dispatch)
[[vk::push_constant]]
uniform DispatchIn dispatchIn;
[shader("compute")]
[numthreads(1, 1, 1)]
void dispatch() {
    dispatchIn.grid[0] = (dispatchIn.flag->flag == true) ?
        (dispatchIn.edges_n + 255)/256 : 0;
    dispatchIn.grid[1] = 1;
    dispatchIn.grid[2] = 1;
    dispatchIn.flag->flag = false;
}

struct PageRankIn{
    uint64_t* row_offsets;
    uint32_t* column_indices;
    float32_t *current_pr;
    float32_t *past_pr;
    uint64_t vertices_n;
    uint64_t edges_n;
};
// @bind(page_rank)
[[vk::push_constant]]
uniform PageRankIn pageRankIn;
[shader("compute")]
[numthreads(256,1,1)]
void page_rank(uint32_t threadId: SV_DispatchThreadID) {
    if (threadId >= pageRankIn.edges_n)
        return;

    uint low = 0;
    uint high = pageRankIn.vertices_n - 1;
    uint src = 0;
    while (low <= high) {
        uint mid = (low + high) / 2;
        if (pageRankIn.row_offsets[mid] <= threadId) {
            src = mid;
            low = mid + 1;
        } else {
            if (mid == 0) break;
            high = mid - 1;
        }
    }

    uint out_degree = pageRankIn.row_offsets[src+1] - pageRankIn.row_offsets[src];
    float to_push = pageRankIn.past_pr[src]/out_degree;
    InterlockedAdd(pageRankIn.current_pr[pageRankIn.column_indices[threadId]], to_push);
}

/* mesh shader */

#define meshWorkGroupSize WAVE_SIZE

struct SharedPRPayload {
    uint vertex_ids[32];
    uint edge_offsets[32+1];
    uint total_edges;
};
groupshared SharedPRPayload sharedPRPayload;

// @bind(pr_task)
[[vk::push_constant]]
uniform PageRankIn prIn;
[shader("amplification")]
[numthreads(WAVE_SIZE, 1, 1)]
void pr_task(
    uint threadId : SV_DispatchThreadID,
    uint lid : SV_GroupIndex,
) {
    uint v = threadId;
    uint degree = 0;
    if (threadId < prIn.vertices_n) {
        degree = uint(prIn.row_offsets[v + 1] - prIn.row_offsets[v]);
    }

    uint wave_offset = WavePrefixSum(degree);

    sharedPRPayload.vertex_ids[lid] = v;
    sharedPRPayload.edge_offsets[lid] = wave_offset;

    if (lid == WaveGetLaneCount() - 1) {
        uint wave_total = wave_offset + degree;
        sharedPRPayload.edge_offsets[32] = wave_total;
        sharedPRPayload.total_edges = wave_total;
    }

    GroupMemoryBarrierWithGroupSync();

    uint mesh_count = (sharedPRPayload.total_edges + meshWorkGroupSize - 1) / meshWorkGroupSize;
    DispatchMesh(mesh_count, 1, 1, sharedPRPayload);
}

[shader("mesh")]
[numthreads(WAVE_SIZE, 1, 1)]
[outputtopology("point")]
void pr_mesh(
    uint threadId : SV_DispatchThreadID,
    uint lid : SV_GroupIndex,
    uint3 gid : SV_GroupID,
    in payload SharedPRPayload inPayload,
    out vertices float4 vertices[1],
    out indices uint indices[1]
) {
    uint src = 0;
    uint local_offset = gid.x * meshWorkGroupSize + lid;
    if(local_offset >= inPayload.total_edges) {
        SetMeshOutputCounts(0, 0);
        return;
    }

    /* evil binary search */
    [unroll]
    for (uint step = (32 >> 1); step > 0; step >>= 1) {
        if (inPayload.edge_offsets[src + step] <= local_offset) {
            src += step;
        }
    }
    uint edge_offset = local_offset - inPayload.edge_offsets[src];
    src = inPayload.vertex_ids[src];
    uint dst = prIn.row_offsets[src] + edge_offset;

    uint out_degree = prIn.row_offsets[src+1] - prIn.row_offsets[src];
    float to_push = prIn.past_pr[src]/out_degree;
    InterlockedAdd(prIn.current_pr[prIn.column_indices[dst]], to_push);

    SetMeshOutputCounts(0, 0);
}
