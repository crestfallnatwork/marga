/* SPDX-License-Identifier: MPL-2.0 */

// Task + mesh shader pair. The task stage dispatches one mesh workgroup per task
// workgroup and passes an element base index through the payload; the mesh stage
// writes one value per element through the BDA output buffer.
//
// Entry points with the same prefix (tile_task + tile_mesh) generate
// tile_shader::TileMesh and the tile_shader::TILEExecutor alias.

struct TileParams {
    uint64_t out_addr;
    uint32_t count;
};

[[vk::push_constant]]
uniform TileParams p;

// Task -> mesh payload. Each task workgroup dispatches exactly one mesh workgroup, so
// base = task group id * mesh workgroup size covers the output range in order.
struct TilePayload {
    uint32_t base;
};

[shader("amplification")]
[numthreads(32, 1, 1)]
void tile_task(uint3 gid : SV_GroupID)
{
    TilePayload payload;
    payload.base = gid.x * 64u;
    DispatchMesh(1, 1, 1, payload);
}

[shader("mesh")]
[numthreads(64, 1, 1)]
[outputtopology("point")]
void tile_mesh(uint lid : SV_GroupIndex, in payload TilePayload payload,
               out vertices float4 vertices[64], out indices uint indices[64])
{
    SetMeshOutputCounts(0, 0);

    const uint32_t index = payload.base + lid;
    if (index < p.count)
        ((uint32_t*)p.out_addr)[index] = index;
}
