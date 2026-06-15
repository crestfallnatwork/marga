// -----------------------------------------------------------------------------
// pagerank.slang
// -----------------------------------------------------------------------------

#define PR_FIXED_POINT_SCALE 1125899906842624ull // 2^50

struct DispatchIndirectCommand {
    uint x, y, z;
};

struct PartialSum {
    uint destination_vertex;
    uint value_low;
    uint value_high;
};

struct TaskPayload {
    uint vertex_ids[32];
    uint edge_offsets[33];
    uint total_edges;
    uint wg_offset; 
};

struct Params {
    uint64_t row_offsets;
    uint64_t column_indices;
    uint64_t out_degrees;
    uint64_t pr_current;
    uint64_t pr_next;
    uint64_t active_vertices_curr;
    uint64_t active_vertices_next;
    uint64_t active_count_curr;
    uint64_t active_count_next;
    uint64_t intermediate;          // Array of PartialSum structs
    uint64_t intermediate_count;    // Pointer to uint (allocator for intermediate array)
    uint64_t indirect_buffer;
    uint64_t dangling_sum;
    
    uint32_t num_vertices;
    uint32_t damping_numerator;
    uint32_t damping_denominator;
    uint32_t base_rank_low;
    uint32_t base_rank_high;
};

[[vk::push_constant]]
uniform Params p;

groupshared TaskPayload sharedPayload;

// =============================================================================
// STAGE 1 & 2: THE SCATTER (TASK & MESH SHADERS)
// =============================================================================

[shader("amplification")]
[numthreads(32, 1, 1)]
void task(
    uint3 threadId : SV_DispatchThreadID,
    uint lid : SV_GroupIndex,
    uint3 groupId : SV_GroupID
) {
    uint current_size = *((uint*)p.active_count_curr);
    uint tid = groupId.x * 32 + lid;

    uint v = 0xFFFFFFFF;
    if (tid < current_size) {
        v = ((uint*)p.active_vertices_curr)[tid];
    }

    uint degree = 0;
    if (v != 0xFFFFFFFF) {
        uint64_t* col_offsets = (uint64_t*)p.row_offsets;
        degree = uint(col_offsets[v + 1] - col_offsets[v]);
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
    
    if (lid == 0 && mesh_count > 0) {
        uint offset;
        // Allocate space for mesh_count * 32 partial sums
        InterlockedAdd(*((uint*)p.intermediate_count), mesh_count * 32, offset);
        sharedPayload.wg_offset = offset;
    }

    GroupMemoryBarrierWithGroupSync();

    DispatchMesh(mesh_count, 1, 1, sharedPayload);
}

groupshared uint local_sums_low[32];
groupshared uint local_sums_high[32];

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
    if (lid < 32) {
        local_sums_low[lid] = 0;
        local_sums_high[lid] = 0;
    }
    GroupMemoryBarrierWithGroupSync();

    uint edge_idx = gid.x * 128 + lid;
    uint64_t pulled_pr = 0;
    uint parent_idx = 0xFFFFFFFF;

    if (edge_idx < inPayload.total_edges) {
        uint low = 0;
        uint high = 31;
        parent_idx = 0;
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

        uint destination_v = inPayload.vertex_ids[parent_idx];
        uint64_t* col_offsets = (uint64_t*)p.row_offsets;
        uint64_t v_start = col_offsets[destination_v];
        uint64_t local_edge_offset = uint64_t(edge_idx - inPayload.edge_offsets[parent_idx]);
        uint64_t global_edge_idx = v_start + local_edge_offset;

        uint u = ((uint32_t*)p.column_indices)[global_edge_idx];
        uint out_deg = ((uint32_t*)p.out_degrees)[u];
        if (out_deg > 0) {
            uint64_t pr_u = ((uint64_t*)p.pr_current)[u];
            pulled_pr = pr_u / uint64_t(out_deg);
        }
    }

    if (pulled_pr > 0 && parent_idx != 0xFFFFFFFF) {
        uint pr_low = uint(pulled_pr & 0xFFFFFFFF);
        uint pr_high = uint(pulled_pr >> 32);

        uint old_low;
        InterlockedAdd(local_sums_low[parent_idx], pr_low, old_low);
        if (old_low + pr_low < old_low) {
            InterlockedAdd(local_sums_high[parent_idx], 1);
        }
        if (pr_high > 0) {
            InterlockedAdd(local_sums_high[parent_idx], pr_high);
        }
    }

    GroupMemoryBarrierWithGroupSync();

    // Write partial sums to intermediate array
    if (lid < 32) {
        uint l_low = local_sums_low[lid];
        uint l_high = local_sums_high[lid];
        uint dest_v = inPayload.vertex_ids[lid];
        
        uint write_idx = inPayload.wg_offset + (gid.x * 32) + lid;
        PartialSum* intermediate = (PartialSum*)p.intermediate;
        intermediate[write_idx].destination_vertex = dest_v;
        intermediate[write_idx].value_low = l_low;
        intermediate[write_idx].value_high = l_high;
    }

    SetMeshOutputCounts(0, 0);
}

// =============================================================================
// STAGE 3: REDUCE INTERMEDIATE BUFFER (COMPUTE)
// =============================================================================

[shader("compute")]
[numthreads(256, 1, 1)]
void reduce(uint32_t threadId : SV_DispatchThreadID) {
    uint total_partials = *((uint*)p.intermediate_count);
    if (threadId >= total_partials) return;

    PartialSum* intermediate = (PartialSum*)p.intermediate;
    PartialSum s = intermediate[threadId];

    if (s.destination_vertex != 0xFFFFFFFF && (s.value_low > 0 || s.value_high > 0)) {
        uint* dest_ptr = (uint*)p.pr_next + (s.destination_vertex * 2);
        uint old_low;
        InterlockedAdd(dest_ptr[0], s.value_low, old_low);
        if (old_low + s.value_low < old_low) {
            InterlockedAdd(dest_ptr[1], 1);
        }
        if (s.value_high > 0) {
            InterlockedAdd(dest_ptr[1], s.value_high);
        }
    }
}

// =============================================================================
// STAGE 4: THE FINALIZE & CONVERGENCE CHECK (COMPUTE)
// =============================================================================

[shader("compute")]
[numthreads(256, 1, 1)]
void finalize_and_check(uint32_t threadId : SV_DispatchThreadID) {
    uint current_size = *((uint*)p.active_count_curr);
    if (threadId >= current_size) return;

    uint v = ((uint*)p.active_vertices_curr)[threadId];
    uint* pr_next_ptr = (uint*)p.pr_next + (v * 2);
    uint64_t* pr_curr = (uint64_t*)p.pr_current;

    uint low = pr_next_ptr[0];
    uint high = pr_next_ptr[1];
    uint64_t gathered_sum = (uint64_t(high) << 32) | uint64_t(low);
    
    uint64_t d_sum = *((uint64_t*)p.dangling_sum);
    uint64_t dangling_contrib = d_sum / uint64_t(p.num_vertices);
    
    uint64_t base_rank = (uint64_t(p.base_rank_high) << 32) | uint64_t(p.base_rank_low);
    
    uint64_t total_sum = gathered_sum + dangling_contrib;
    uint64_t damped_sum = (total_sum * uint64_t(p.damping_numerator)) / uint64_t(p.damping_denominator);
    uint64_t final_pr = base_rank + damped_sum;
    
    pr_curr[v] = final_pr;
    pr_next_ptr[0] = 0;
    pr_next_ptr[1] = 0;
}

// =============================================================================
// STAGE 5: DANGLING NODE SUM (COMPUTE)
// =============================================================================

groupshared uint local_dangling_low;
groupshared uint local_dangling_high;

[shader("compute")]
[numthreads(256, 1, 1)]
void calc_dangling(uint32_t threadId : SV_DispatchThreadID, uint32_t lid : SV_GroupIndex) {
    if (lid == 0) {
        local_dangling_low = 0;
        local_dangling_high = 0;
    }
    GroupMemoryBarrierWithGroupSync();

    uint64_t my_val = 0;
    if (threadId < p.num_vertices) {
        uint out_deg = ((uint32_t*)p.out_degrees)[threadId];
        if (out_deg == 0) {
            my_val = ((uint64_t*)p.pr_current)[threadId];
        }
    }

    if (my_val > 0) {
        uint low = uint(my_val & 0xFFFFFFFF);
        uint high = uint(my_val >> 32);
        
        uint old_low;
        InterlockedAdd(local_dangling_low, low, old_low);
        if (old_low + low < old_low) {
            InterlockedAdd(local_dangling_high, 1);
        }
        if (high > 0) {
            InterlockedAdd(local_dangling_high, high);
        }
    }

    GroupMemoryBarrierWithGroupSync();

    if (lid == 0 && (local_dangling_low > 0 || local_dangling_high > 0)) {
        uint* global_dang = (uint*)p.dangling_sum;
        uint old_low;
        InterlockedAdd(global_dang[0], local_dangling_low, old_low);
        if (old_low + local_dangling_low < old_low) {
            InterlockedAdd(global_dang[1], 1);
        }
        if (local_dangling_high > 0) {
            InterlockedAdd(global_dang[1], local_dangling_high);
        }
    }
}
