/* SPDX-License-Identifier: MPL-2.0 */

// Deliberately invalid: a PICTOR_-prefixed guarded default must be a literal, and
// pictor_reflect rejects this at generation time. Never add this to pictor_add_shader;
// the ctest target invokes the reflector directly and asserts the diagnostic.

#ifndef PICTOR_BAD
#define PICTOR_BAD (1 << 3)
#endif

[shader("compute")]
[numthreads(1, 1, 1)]
void bad_tunable_compute(uint3 tid : SV_DispatchThreadID)
{
    ((uint*)0)[tid.x] = PICTOR_BAD;
}
