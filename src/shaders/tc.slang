// SPDX-License-Identifier: GPL-2.0-only
// Author: Himanshu Gupta <himanshu3@iisc.ac.in>

static const uint cMaxMeshDim = 65535;

T At<T>(T* ptr, uint64_t offset) {
    return *(T*)(uint64_t(ptr) + sizeof(T) * offset);
}
T* Splice<T>(T* ptr, uint64_t offset) {
    return (T*)(uint64_t(ptr) + sizeof(T) * offset);
}

struct SharedTCPayload {
    uint offsets[WAVE_SIZE+1];
    uint u[WAVE_SIZE];
    uint64_t edge_start;
};
groupshared SharedTCPayload tcPayload;

struct TCIn {
    uint64_t* list_offset;
    uint32_t* edge_lists;
    uint32_t* tc;
    uint64_t num_vertices;
    uint64_t edge_start;
    uint64_t edge_num;
};
// @bind(tc_task)
[[vk::push_constant]]
uniform TCIn tcIn;
[shader("amplification")]
[numthreads(WAVE_SIZE, 1, 1)]
void tc_task(
    uint3 gid : SV_GroupID,
    uint lid : SV_GroupIndex,
) {
    uint gid = (gid.y*cMaxMeshDim) + gid.x;
    uint64_t eid = gid*WAVE_SIZE + lid + tcIn.edge_start;

    uint32_t u = 0;
    if(WaveIsFirstLane()) {
        tcPayload.edge_start = eid;
        uint low = 0;
        uint high = tcIn.num_vertices;
        while (low < high) {
            uint mid = low + (high - low) / 2;
            if (At(tcIn.list_offset, mid) > eid) {
                high = mid;
            } else {
                low = mid + 1;
            }
        }
        u = low - 1;
    }

    uint candidates = 0;
    uint64_t num_edges = tcIn.edge_start + tcIn.edge_num;
    if(eid < num_edges) {
        u = WaveReadLaneFirst(u);
        while(eid >= tcIn.list_offset[u+1]) {
            u++;
        }
        uint v = At(tcIn.edge_lists, eid);
        if(u < v) {
            uint n_v = tcIn.list_offset[v+1] - tcIn.list_offset[v];
            uint n_u = tcIn.list_offset[u+1] - tcIn.list_offset[u];
            candidates = min(n_v, n_u);
        }
    }

    tcPayload.u[lid] = u;
    uint candidates_scan = WavePrefixSum(candidates);
    if(lid == WAVE_SIZE - 1) {
        tcPayload.offsets[WAVE_SIZE] = candidates_scan + candidates;
    }
    tcPayload.offsets[lid] = candidates_scan;

    GroupMemoryBarrierWithGroupSync();

    uint mesh_count = (tcPayload.offsets[WAVE_SIZE] + 127) / 128;
    uint mesh_x = mesh_count > cMaxMeshDim ? cMaxMeshDim : mesh_count;
    uint mesh_y = mesh_count > cMaxMeshDim ?
                  ((mesh_count + cMaxMeshDim - 1) / cMaxMeshDim): 1;

    DispatchMesh(mesh_x, mesh_y, 1, tcPayload);
}

[shader("mesh")]
[numthreads(128, 1, 1)]
[outputtopology("point")]
void tc_mesh(
    uint lid: SV_GroupIndex,
    uint3 gid: SV_GroupID,
    in payload SharedTCPayload inPayload,
    out vertices float4 vertices[1],
    out indices uint indices[1]
) {
    uint gid = (gid.y*cMaxMeshDim) + gid.x;
    uint threadId = gid*128 + lid;
    if(threadId >= inPayload.offsets[WAVE_SIZE]) {
        SetMeshOutputCounts(0, 0);
        return;
    }
    uint local_eid = 0;
    [unroll]
    for (uint step = (WAVE_SIZE >> 1); step > 0; step >>= 1) {
        if (inPayload.offsets[local_eid + step] <= threadId) {
            local_eid += step;
        }
    }

    uint64_t s_offset;
    uint64_t b_offset;
    uint32_t b_size;

    uint64_t global_eid = inPayload.edge_start + local_eid;
    uint u = inPayload.u[local_eid];
    uint v = At(tcIn.edge_lists, global_eid);
    uint n_v = tcIn.list_offset[v+1] - tcIn.list_offset[v];
    uint n_u = tcIn.list_offset[u+1] - tcIn.list_offset[u];
    if(n_u < n_v) {
        s_offset = tcIn.list_offset[u];
        b_offset = tcIn.list_offset[v];
        b_size = n_v;
    } else {
        s_offset = tcIn.list_offset[v];
        b_offset = tcIn.list_offset[u];
        b_size = n_u;
    }

    uint64_t local_w_idx = threadId - inPayload.offsets[local_eid];
    uint w = At(tcIn.edge_lists, local_w_idx + s_offset);
    if(w > v) {
        uint low = 0;
        uint high = b_size;
        while(low < high) {
            uint mid = low + (high-low)/2;
            uint mid_val = At(tcIn.edge_lists, mid + b_offset);
            if(mid_val < w) {
                low = mid + 1;
            } else if(w < mid_val) {
                high = mid;
            } else {
                InterlockedAdd(*tcIn.tc, 1);
                break;
            }
        }
    }

    SetMeshOutputCounts(0, 0);
    return;
}
