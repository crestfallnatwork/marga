/* SPDX-License-Identifier: MPL-2.0 */

// B1/B2 regression fixture: only the live guarded default may be extracted. The
// commented-out guarded defaults (a conflicting value and a non-literal one), the
// #if 0 block and the false #ifdef block must all be ignored. Never add this to
// pictor_add_shader; the ctest target runs the reflector and inspects --dump.

#ifndef PICTOR_LIVE
#define PICTOR_LIVE 3
#endif

/*
#ifndef PICTOR_LIVE
#define PICTOR_LIVE 12
#endif
#ifndef PICTOR_COMMENTED
#define PICTOR_COMMENTED (1 << 3)
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

[shader("compute")]
[numthreads(1, 1, 1)]
void ghost_compute(uint3 tid : SV_DispatchThreadID)
{
    ((uint*)0)[tid.x] = PICTOR_LIVE;
}
