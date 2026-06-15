
// Standard Vulkan Indirect Dispatch Layout
struct DispatchIndirectCommand {
    uint x, y, z;
};

// The payload passed from Task to Mesh shader
struct TaskPayload {
    uint vertex_ids[32];
    uint edge_offsets[33];
    uint total_edges;
};

// All parameters passed via Push Constants (Pure BDA)
struct Params {
    uint32_t unused;
    uint32_t current_level;
    uint64_t row_offsets;
    uint64_t column_indices;
    uint64_t visited;
    uint64_t distances;
    uint64_t current_frontier;
    uint64_t next_frontier;
    uint64_t current_frontier_size;
    uint64_t next_frontier_size;
    uint64_t indirect_buffer;
    uint64_t num_vertices;
};

[[vk::push_constant]]
uniform Params p;

groupshared TaskPayload sharedPayload;

// --- Entry Point: Indirect Dispatch Preparation ---
[shader("compute")]
[numthreads(1, 1, 1)]
void compute_indirect() {
    uint* size_ptr = (uint*)p.next_frontier_size;
    uint size = *size_ptr;
    
    uint total_groups = (size + 31) / 32;
    
    DispatchIndirectCommand* cmd = (DispatchIndirectCommand*)p.indirect_buffer;
    
    if (total_groups <= 65535) {
        cmd->x = total_groups;
        cmd->y = 1;
    } else {
        cmd->x = 65535;
        cmd->y = (total_groups + 65534) / 65535;
    }
    cmd->z = 1;
}

// --- Entry Point: Task/Amplification Shader ---
[shader("amplification")]
[numthreads(32, 1, 1)]
void task(
    uint3 threadId : SV_DispatchThreadID,
    uint lid : SV_GroupIndex,
    uint3 groupId : SV_GroupID
) {
    uint current_size = *((uint*)p.current_frontier_size);
    uint tid = groupId.y * (65535 * 32) + threadId.x;

    uint v = 0xFFFFFFFF;
    if (tid < current_size) {
        v = ((uint*)p.current_frontier)[tid];
    }

    uint degree = 0;
    if (v != 0xFFFFFFFF) {
        uint64_t* row_offsets = (uint64_t*)p.row_offsets;
        degree = uint(row_offsets[v + 1] - row_offsets[v]);
    }

    uint wave_offset = WavePrefixSum(degree);
    uint wave_total = WaveActiveSum(degree);

    sharedPayload.vertex_ids[lid] = v;
    sharedPayload.edge_offsets[lid] = wave_offset;
    
    if (lid == 31) {
        sharedPayload.edge_offsets[32] = wave_offset + degree;
        sharedPayload.total_edges = wave_total;
    }

    GroupMemoryBarrierWithGroupSync();

    uint mesh_count = (sharedPayload.total_edges + 127) / 128;
    DispatchMesh(mesh_count, 1, 1, sharedPayload);
}

// --- Entry Point: Mesh Shader ---
[shader("mesh")]
[numthreads(128, 1, 1)]
[outputtopology("point")]
void mesh(
    uint lid : SV_GroupIndex,
    uint3 gid : SV_GroupID,
    in payload TaskPayload inPayload,
    out vertices float4 vertices[1],
    out indices uint indices[1]
) {
    uint edge_idx = gid.x * 128 + lid;
    
    bool added = false;
    uint neighbor = 0xFFFFFFFF;

    if (edge_idx < inPayload.total_edges) {
        uint low = 0;
        uint high = 31;
        uint parent_idx = 0;
        while (low <= high) {
            uint mid = (low + high) / 2;
            if (inPayload.edge_offsets[mid] <= edge_idx) {
                parent_idx = mid;
                low = mid + 1;
            } else {
                if (mid == 0) break;
                high = mid - 1;
            }
        }

        uint u = inPayload.vertex_ids[parent_idx];
        uint64_t* rows = (uint64_t*)p.row_offsets;
        uint64_t u_start = rows[u];
        uint64_t local_edge_offset = uint64_t(edge_idx - inPayload.edge_offsets[parent_idx]);
        uint64_t global_edge_idx = u_start + local_edge_offset;

        neighbor = ((uint32_t*)p.column_indices)[global_edge_idx];

        uint word_idx = neighbor / 32;
        uint bit_idx = neighbor % 32;
        uint mask = 1 << bit_idx;

        uint32_t* visited = (uint32_t*)p.visited;
        
        if ((visited[word_idx] & mask) == 0) {
            uint old_val;
            InterlockedOr(visited[word_idx], mask, old_val);

            if ((old_val & mask) == 0) {
                added = true;
                ((uint32_t*)p.distances)[neighbor] = p.current_level + 1;
            }
        }
    }

    uint wave_count = WaveActiveCountBits(added);
    uint wave_offset = WavePrefixCountBits(added);
    
    uint global_base_pos = 0;
    if (WaveIsFirstLane()) {
        InterlockedAdd(*((uint*)p.next_frontier_size), wave_count, global_base_pos);
    }
    global_base_pos = WaveReadLaneFirst(global_base_pos);
    
    if (added) {
        ((uint*)p.next_frontier)[global_base_pos + wave_offset] = neighbor;
    }

    SetMeshOutputCounts(0, 0);
}
