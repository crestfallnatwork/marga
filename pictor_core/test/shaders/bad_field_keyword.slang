/* SPDX-License-Identifier: MPL-2.0 */

// Deliberately invalid: PICTOR_FOR maps to the C++ keyword 'for', so pictor_reflect
// rejects it at generation time. Never add this to pictor_add_shader; the ctest target
// invokes the reflector directly and expects the diagnostic.

#ifndef PICTOR_FOR
#define PICTOR_FOR 1
#endif

[shader("compute")]
[numthreads(1, 1, 1)]
void bad_field_keyword_compute(uint3 tid : SV_DispatchThreadID)
{
    ((uint*)0)[tid.x] = PICTOR_FOR;
}
