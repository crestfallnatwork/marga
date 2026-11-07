/* SPDX-License-Identifier: MPL-2.0 */

// Tuner test shader. PICTOR_THREADS participates in [numthreads], so a tuner that overrides it
// changes the reflected workgroup size; PICTOR_BIAS changes the computed value. The compute
// family and the task/mesh family share the tunables, so both tuned executor factories are
// exercised.

#ifndef PICTOR_THREADS
#define PICTOR_THREADS 32
#endif

#ifndef PICTOR_BIAS
#define PICTOR_BIAS 1
#endif

struct TunerParams {
    uint64_t out_addr;
    uint32_t count;
};

[[vk::push_constant]]
uniform TunerParams p;

[shader("compute")]
[numthreads(PICTOR_THREADS, 1, 1)]
void tuner_compute(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= p.count)
        return;

    ((uint*)p.out_addr)[tid.x] = tid.x + PICTOR_BIAS;
}

struct TunerMeshPayload {
    uint32_t base;
};

[shader("amplification")]
[numthreads(PICTOR_THREADS, 1, 1)]
void tuner_task(uint3 gid : SV_GroupID)
{
    TunerMeshPayload payload;
    payload.base = gid.x * PICTOR_THREADS;
    DispatchMesh(1, 1, 1, payload);
}

[shader("mesh")]
[numthreads(PICTOR_THREADS, 1, 1)]
[outputtopology("point")]
void tuner_mesh(uint lid : SV_GroupIndex, in payload TunerMeshPayload payload,
                out vertices float4 vertices[32], out indices uint indices[32])
{
    SetMeshOutputCounts(0, 0);

    const uint32_t index = payload.base + lid;
    if (index < p.count)
        ((uint*)p.out_addr)[index] = index + PICTOR_BIAS;
}
