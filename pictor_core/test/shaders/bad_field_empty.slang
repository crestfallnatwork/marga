/* SPDX-License-Identifier: MPL-2.0 */

// Deliberately invalid: PICTOR_ strips to the empty field name. Never add this to
// pictor_add_shader; the ctest target invokes the reflector directly and expects the
// diagnostic.

#ifndef PICTOR_
#define PICTOR_ 5
#endif

[shader("compute")]
[numthreads(1, 1, 1)]
void bad_field_empty_compute(uint3 tid : SV_DispatchThreadID)
{
    ((uint*)0)[tid.x] = PICTOR_;
}
