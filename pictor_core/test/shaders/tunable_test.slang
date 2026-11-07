/* SPDX-License-Identifier: MPL-2.0 */

// Tunable test shader: every live PICTOR_-prefixed guarded literal default is a
// shader-declared tunable the reflector extracts into tunable_test_shader::TunableTestTunables.
// PICTOR_MODULE_BIAS is declared in the imported module (tunable_ops), exercising the
// whole-closure scan. The ghost blocks below (a multi-line comment, #if 0 and a false
// #ifdef) must never produce tunables; a regression either grows defines() or turns the
// commented-out conflicting default into a generation error.

import tunable_ops;

#ifndef PICTOR_SCALE
#define PICTOR_SCALE 3
#endif

#ifndef PICTOR_OFFSET
#define PICTOR_OFFSET 1
#endif

#ifndef PICTOR_ENABLE_EXTRA
#define PICTOR_ENABLE_EXTRA false
#endif

#ifndef PICTOR_OCTAL
#define PICTOR_OCTAL 010
#endif

#define PICTOR_LIVE 10

/*
#ifndef PICTOR_SCALE
#define PICTOR_SCALE 12
#endif
#ifndef PICTOR_COMMENTED
#define PICTOR_COMMENTED (1 << 3)
#endif
#ifndef PICTOR_LIVE
#define PICTOR_LIVE 12
#endif
*/

#if 0
#ifndef PICTOR_DEAD
#define PICTOR_DEAD 7
#endif
#endif

#ifdef NEVER_DEFINED_MACRO
#ifndef PICTOR_DEAD_IFDEF
#define PICTOR_DEAD_IFDEF 9
#endif
#endif

struct TunableParams {
    uint64_t out_addr;
    uint32_t count;
};

[[vk::push_constant]]
uniform TunableParams p;

[shader("compute")]
[numthreads(32, 1, 1)]
void tunable_compute(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= p.count)
        return;

    uint value = tid.x * PICTOR_SCALE + PICTOR_OFFSET + module_bias() + PICTOR_OCTAL;
    if (PICTOR_ENABLE_EXTRA)
        value += 100u;

    ((uint*)p.out_addr)[tid.x] = value;
}

struct TunableMeshPayload {
    uint32_t base;
};

[shader("amplification")]
[numthreads(32, 1, 1)]
void tunable_task(uint3 gid : SV_GroupID)
{
    TunableMeshPayload payload;
    payload.base = gid.x * 32u;
    DispatchMesh(1, 1, 1, payload);
}

[shader("mesh")]
[numthreads(32, 1, 1)]
[outputtopology("point")]
void tunable_mesh(uint lid : SV_GroupIndex, in payload TunableMeshPayload payload,
                  out vertices float4 vertices[32], out indices uint indices[32])
{
    SetMeshOutputCounts(0, 0);

    const uint32_t index = payload.base + lid;
    if (index < p.count)
        ((uint*)p.out_addr)[index] = index * PICTOR_SCALE + PICTOR_OFFSET;
}
