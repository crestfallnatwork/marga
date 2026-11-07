/* SPDX-License-Identifier: MPL-2.0 */

// Deliberately invalid: PICTOR_1 strips to the field name '1', which is not a C++
// identifier. Never add this to pictor_add_shader; the ctest target invokes the
// reflector directly and expects the diagnostic.

#ifndef PICTOR_1
#define PICTOR_1 6
#endif

[shader("compute")]
[numthreads(1, 1, 1)]
void bad_field_digit_compute(uint3 tid : SV_DispatchThreadID)
{
    ((uint*)0)[tid.x] = PICTOR_1;
}
