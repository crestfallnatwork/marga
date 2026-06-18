struct SharedBFSPayload {
    uint vertex_ids[WAVE_SIZE];
    uint edge_offsets[WAVE_SIZE+1];
    uint total_edges;
};
groupshared SharedBFSPayload sharedBFSPayload;

struct BFSIn {
    uint32_t current_level;
    uint64_t* row_offsets;
    uint32_t* column_indices;
    uint32_t* visited;
    uint32_t* distances;
    uint32_t* current_frontier;
    uint32_t* next_frontier;
    uint32_t* current_frontier_size;
    uint32_t* next_frontier_size;
    uint64_t num_vertices;
};
static const uint cMaxMeshDim = 65535;
// @bind(bfs_task)
[[vk::push_constant]]
uniform BFSIn bfsIn;
[shader("amplification")]
[numthreads(WAVE_SIZE, 1, 1)]
void bfs_task(
    uint3 gid : SV_GroupID,
    uint lid : SV_GroupIndex
) {
    uint gId = (gid.y*cMaxMeshDim) + gid.x;
    uint threadId = gId*WAVE_SIZE + lid;

    uint v = 0xFFFFFFFF;
    uint current_size = *bfsIn.current_frontier_size;
    if (threadId < current_size) {
        v = bfsIn.current_frontier[threadId];
    }

    uint degree = 0;
    if (v != 0xFFFFFFFF) {
        degree = uint(bfsIn.row_offsets[v + 1] - bfsIn.row_offsets[v]);
    }

    uint wave_offset = WavePrefixSum(degree);
    uint wave_total = WaveActiveSum(degree);

    sharedBFSPayload.vertex_ids[lid] = v;
    sharedBFSPayload.edge_offsets[lid] = wave_offset;
    
    if (lid == WAVE_SIZE - 1) {
        sharedBFSPayload.edge_offsets[WAVE_SIZE] = wave_offset + degree;
        sharedBFSPayload.total_edges = wave_total;
    }

    GroupMemoryBarrierWithGroupSync();
    uint mesh_count = (sharedBFSPayload.total_edges + 127) / 128;

    uint mesh_x = mesh_count > cMaxMeshDim ? cMaxMeshDim : mesh_count;
    uint mesh_y = mesh_count > cMaxMeshDim ?
                  ((mesh_count + cMaxMeshDim - 1) / cMaxMeshDim): 1;

    DispatchMesh(mesh_x, mesh_y, 1, sharedBFSPayload);
}

[shader("mesh")]
[numthreads(128, 1, 1)]
[outputtopology("point")]
void bfs_mesh(
    uint lid : SV_GroupIndex,
    uint3 gid : SV_GroupID,
    in payload SharedBFSPayload inPayload,
    out vertices float4 vertices[1],
    out indices uint indices[1]
) {
    uint gId = (gid.y*cMaxMeshDim) + gid.x;
    uint edge_idx = gId*128 + lid;

    if(edge_idx >= inPayload.total_edges) {
        SetMeshOutputCounts(0, 0);
        return;
    }

    bool added = false;
    uint neighbor = 0xFFFFFFFF;
    uint parent_idx = 0;

    /* evil binary search */
    [unroll]
    for (uint step = (WAVE_SIZE >> 1); step > 0; step >>= 1) {
        if (inPayload.edge_offsets[parent_idx + step] <= edge_idx) {
            parent_idx += step;
        }
    }

    uint u = inPayload.vertex_ids[parent_idx];
    uint64_t u_start = bfsIn.row_offsets[u];
    uint64_t local_edge_offset = edge_idx - inPayload.edge_offsets[parent_idx];
    uint64_t global_edge_idx = u_start + local_edge_offset;

    uint64_t byte_addr = uint64_t(bfsIn.column_indices) + (global_edge_idx * 4ull);
    neighbor = *(uint32_t*)byte_addr;

    uint word_idx = neighbor / 32;
    uint bit_idx = neighbor % 32;
    uint mask = 1 << bit_idx;

    if ((bfsIn.visited[word_idx] & mask) == 0) {
        uint old_val;
        InterlockedOr(bfsIn.visited[word_idx], mask, old_val);

        if ((old_val & mask) == 0) {
            added = true;
            bfsIn.distances[neighbor] = bfsIn.current_level + 1;
        }
    }

    uint wave_count = WaveActiveCountBits(added);
    uint wave_offset = WavePrefixCountBits(added);
    
    uint global_base_pos = 0;
    if (WaveIsFirstLane()) {
        InterlockedAdd(*bfsIn.next_frontier_size, wave_count, global_base_pos);
    }
    global_base_pos = WaveReadLaneFirst(global_base_pos);
    
    if (added) {
        bfsIn.next_frontier[global_base_pos + wave_offset] = neighbor;
    }

    SetMeshOutputCounts(0, 0);
}

struct DispatchIn {
    uint32_t *frontier_size;
    uint32_t *grid;
};
// @bind(dispatch)
[[vk::push_constant]]
uniform DispatchIn dispatchIn;
[shader("compute")]
[numthreads(1,1,1)]
void dispatch() {
    uint32_t frontier_size = *dispatchIn.frontier_size;
    uint32_t *grid = dispatchIn.grid;
    uint32_t task_count = (frontier_size + WAVE_SIZE - 1)/WAVE_SIZE;
    if(task_count > cMaxMeshDim) {
        grid[0] = cMaxMeshDim;
        grid[1] = (task_count + cMaxMeshDim - 1)/cMaxMeshDim;
    } else {
        grid[0] = task_count;
        grid[1] = 1;
    }
    grid[2] = 1;
}

/* Vertex based compute shader. Sanity check
 * that this arch indeed makes sense and is
 * strictly better for sparse power law graphs
 * On RTX 2070:
 * GAP-twitter: ~8x faster
 * soc-orkut: ~3x faster
 */
// @bind(compute_bfs)
[[vk::push_constant]]
uniform BFSIn bfsInCompute;
[shader("compute")]
[numthreads(128, 1, 1)]
void compute_bfs(uint3 tid : SV_DispatchThreadID) {
    uint current_size = *bfsInCompute.current_frontier_size;
    uint v = 0xFFFFFFFF;

    if (tid.x < current_size) {
        v = bfsInCompute.current_frontier[tid.x];
    }

    uint64_t start = (v != 0xFFFFFFFF) ? bfsInCompute.row_offsets[v] : 0;
    uint64_t end   = (v != 0xFFFFFFFF) ? bfsInCompute.row_offsets[v+1] : 0;
    uint64_t edge_iter = start;

    while (WaveActiveAnyTrue(edge_iter < end)) {
        bool process_edge = (edge_iter < end);
        uint neighbor = 0xFFFFFFFF;
        bool added = false;

        if (process_edge) {
            uint64_t byte_addr = uint64_t(bfsInCompute.column_indices) + (edge_iter * 4ull);
            neighbor = *(uint32_t*)byte_addr;
            edge_iter++;

            uint word_idx = neighbor / 32;
            uint bit_idx = neighbor % 32;
            uint mask = 1 << bit_idx;

            if ((bfsInCompute.visited[word_idx] & mask) == 0) {
                uint old_val;
                InterlockedOr(bfsInCompute.visited[word_idx], mask, old_val);

                if ((old_val & mask) == 0) {
                    added = true;
                    bfsInCompute.distances[neighbor] = bfsInCompute.current_level + 1;
                }
            }
        }

        uint wave_count = WaveActiveCountBits(added);
        uint wave_offset = WavePrefixCountBits(added);

        uint global_base_pos = 0;
        
        if (WaveIsFirstLane() && wave_count > 0) {
            InterlockedAdd(*bfsInCompute.next_frontier_size, wave_count, global_base_pos);
        }
        
        global_base_pos = WaveReadLaneFirst(global_base_pos);

        if (added) {
            bfsInCompute.next_frontier[global_base_pos + wave_offset] = neighbor;
        }
    }
}
