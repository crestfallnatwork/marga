/* SPDX-License-Identifier: MPL-2.0 */

// A task/mesh pair must resolve to one push-constant layout; this pair deliberately does not.
// pictor_reflect must reject it instead of choosing one stage's layout silently.

struct TaskIn {
    uint32_t value;
};

[[vk::push_constant]]
uniform TaskIn task_in;

struct MeshIn {
    uint32_t* out;
};

[[vk::push_constant]]
uniform MeshIn mesh_in;

struct MismatchPayload {
    uint32_t base;
};

[shader("amplification")]
[numthreads(1, 1, 1)]
void mismatch_task(uint3 gid : SV_GroupID)
{
    MismatchPayload payload;
    payload.base = task_in.value + gid.x;
    DispatchMesh(1, 1, 1, payload);
}

[shader("mesh")]
[numthreads(1, 1, 1)]
[outputtopology("point")]
void mismatch_mesh(uint lid : SV_GroupIndex, in payload MismatchPayload payload,
                   out vertices float4 vertices[1], out indices uint indices[1])
{
    SetMeshOutputCounts(0, 0);
    if (lid == 0 && payload.base != 0xFFFFFFFFu)
        *mesh_in.out = payload.base;
}
