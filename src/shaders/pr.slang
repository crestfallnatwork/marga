static const uint cMaxMeshDim = 65535;

struct ReductionFlag{
    uint32_t iterations;
    uint32_t padding[30];
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

struct PageRankCSCIn{
    uint64_t* row_offsets;
    uint32_t* column_indices;
    float32_t *past_pr;
    float32_t *curr_pr;
    uint32_t *out_degrees;
    uint64_t vertices_n;
    uint64_t edges_n;
};

struct SharedPRCPayload {
    uint v_start;
    uint v_end;
};
groupshared SharedPRCPayload sharedPRPayload;

// @bind(prc_task)
[[vk::push_constant]]
uniform PageRankCSCIn prIn;
[shader("amplification")]
[numthreads(1, 1, 1)]
void prc_task(
    uint threadId : SV_DispatchThreadID,
    uint lid : SV_GroupIndex,
) {

    uint32_t v_start = threadId * 64;
    uint32_t v_end = v_start + 64;
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
        for (uint step = (64 >> 1); step > 0; step >>= 1) {
            if (dst + step < prIn.vertices_n && prIn.row_offsets[dst + step] <= edge_offset) {
                dst += step;
            }
        }

        uint src = *(prIn.column_indices + edge_offset);
        pull = prIn.past_pr[src]/prIn.out_degrees[src];
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
    uint32_t *start_col = outDegreeIn.column_indices + outDegreeIn.row_offsets[threadId];
    for(uint32_t i = 0; i < degree; i++) {
        uint32_t src = *(start_col + i);
        InterlockedAdd(outDegreeIn.out_degrees[src], 1);
    }
}

struct DispatchIn {
    ReductionFlag *flag;
    uint* mesh_grid;
    uint* reduce_grid;
    uint vertices_n;
}
// @bind(dispatch)
[[vk::push_constant]]
uniform DispatchIn dispatchIn;
[numthreads(1,1,1)]
[shader("compute")]
void dispatch() {
    uint* mesh_grid = dispatchIn.mesh_grid;
    uint* reduce_grid = dispatchIn.reduce_grid;
    if(dispatchIn.flag->flag) {
        dispatchIn.flag->iterations++;
        mesh_grid[0] = ((dispatchIn.vertices_n + 63)/64);
        reduce_grid[0] = ((dispatchIn.vertices_n + 255)/256);
    } else {
        mesh_grid[0] = 0;
        reduce_grid[0] = 0;
    }

    mesh_grid[1] = 1;
    reduce_grid[1] = 1;
    mesh_grid[2] = 1;
    reduce_grid[2] = 1;
    dispatchIn.flag->flag = false;
}
