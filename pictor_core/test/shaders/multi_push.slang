/* SPDX-License-Identifier: MPL-2.0 */

// Regression: a module with several push-constant blocks. Every block is declared before the
// entries, and the entries are ordered differently from the blocks, so neither "last block",
// "nearest preceding block" nor "entry index equals block index" can produce the right answer.

struct AlphaIn {
    uint32_t* out;
    uint32_t  value;
};

struct BetaIn {
    uint32_t* out;
    uint32_t  value;
};

struct GammaIn {
    uint32_t* out;
    uint32_t  value;
};

[[vk::push_constant]]
uniform AlphaIn alpha;

[[vk::push_constant]]
uniform BetaIn beta;

[[vk::push_constant]]
uniform GammaIn gamma;

struct GammaPayload {
    uint32_t base;
};

[shader("amplification")]
[numthreads(1, 1, 1)]
void gamma_task(uint3 gid : SV_GroupID)
{
    GammaPayload payload;
    payload.base = gid.x * (gamma.value ? 1u : 0u);
    DispatchMesh(1, 1, 1, payload);
}

[shader("mesh")]
[numthreads(1, 1, 1)]
[outputtopology("point")]
void gamma_mesh(uint lid : SV_GroupIndex, in payload GammaPayload payload,
                out vertices float4 vertices[1], out indices uint indices[1])
{
    SetMeshOutputCounts(0, 0);
    if (lid == 0 && payload.base == 0)
        *gamma.out = gamma.value;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void beta_compute(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x == 0 && tid.y == 0 && tid.z == 0)
        *beta.out = beta.value;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void alpha_compute(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x == 0 && tid.y == 0 && tid.z == 0)
        *alpha.out = alpha.value;
}

groupshared uint delta_scratch;

[shader("compute")]
[numthreads(1, 1, 1)]
void delta_compute(uint3 tid : SV_DispatchThreadID)
{
    delta_scratch = tid.x;
}
