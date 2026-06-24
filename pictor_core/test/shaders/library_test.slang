/* SPDX-License-Identifier: MPL-2.0 */
/* Author: Himanshu Gupta <himanshu3@iisc.ac.in> */

struct AddParams {
    uint64_t a_addr;
    uint64_t b_addr;
    uint64_t c_addr;
    uint32_t count;
};

struct SimpleParams {
    uint64_t out_addr;
    uint32_t count;
};

struct AllParams {
    AddParams add;
    SimpleParams simple;
};

[[vk::push_constant]]
uniform AllParams p;

[shader("compute")]
[numthreads(64, 1, 1)]
void add_compute(uint3 threadId : SV_DispatchThreadID) {
    if (threadId.x < p.add.count) {
        float* a = (float*)p.add.a_addr;
        float* b = (float*)p.add.b_addr;
        float* c = (float*)p.add.c_addr;
        c[threadId.x] = a[threadId.x] + b[threadId.x];
    }
}

struct SimplePayload {
    float values[64];
};

groupshared SimplePayload shared_payload;

[shader("amplification")]
[numthreads(32, 1, 1)]
void simple_task(uint tid : SV_GroupIndex) {
    shared_payload.values[tid] = float(tid) * 2.0f;
    shared_payload.values[tid + 32] = float(tid + 32) * 2.0f;
    
    GroupMemoryBarrierWithGroupSync();
    if (tid == 0) {
        DispatchMesh(1, 1, 1, shared_payload);
    }
}

[shader("mesh")]
[numthreads(64, 1, 1)]
[outputtopology("point")]
void simple_mesh(
    uint lid : SV_GroupIndex,
    in payload SimplePayload payload,
    out vertices float4 vertices[64],
    out indices uint indices[64]
) {
    if (lid < p.simple.count) {
        float* out = (float*)p.simple.out_addr;
        out[lid] = payload.values[lid];
    }
    SetMeshOutputCounts(0, 0);
}
