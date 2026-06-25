// SPDX-License-Identifier: GPL-2.0-only
// Author: Himanshu Gupta <himanshu3@iisc.ac.in>

static const uint cMaxMeshDim = 65535;

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
    uint v_start;
    uint v_end;
};
groupshared SharedPRPayload sharedPRPayload;

// @bind(pr_task)
[[vk::push_constant]]
uniform PageRankIn prIn;
[shader("amplification")]
[numthreads(1, 1, 1)]
void pr_task(
    uint threadId : SV_DispatchThreadID,
    uint lid : SV_GroupIndex,
) {

    uint32_t v_start = threadId * 32;
    uint32_t v_end = (threadId + 1) * 32;
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
[numthreads(32, 1, 1)]
[outputtopology("point")]
void pr_mesh(
    uint threadId : SV_DispatchThreadID,
    uint lid : SV_GroupIndex,
    uint3 gid : SV_GroupID,
    in payload SharedPRPayload inPayload,
    out vertices float4 vertices[1],
    out indices uint indices[1]
) {
    uint gid_flat = gid.y * cMaxMeshDim + gid.x;
    uint local_offset = gid_flat * meshWorkGroupSize + lid;

    uint32_t v_start = inPayload.v_start;
    uint32_t v_end = inPayload.v_end;

    uint32_t total_edges = prIn.row_offsets[v_end] - prIn.row_offsets[v_start];
    if(local_offset >= total_edges) {
        SetMeshOutputCounts(0, 0);
        return;
    }

    uint32_t src = v_start;
    uint64_t edge_offset = local_offset + prIn.row_offsets[src];
    /* evil binary search */
    [unroll]
    for (uint step = (32 >> 1); step > 0; step >>= 1) {
        if (prIn.row_offsets[src + step] <= edge_offset) {
            src += step;
        }
    }
    uint dst = *(prIn.column_indices + edge_offset);

    uint out_degree = prIn.row_offsets[src+1] - prIn.row_offsets[src];
    float to_push = prIn.past_pr[src]/out_degree;
    InterlockedAdd(prIn.current_pr[dst], to_push);

    SetMeshOutputCounts(0, 0);
}

