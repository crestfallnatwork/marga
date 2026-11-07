/* SPDX-License-Identifier: MPL-2.0 */

// Deliberately invalid: PICTOR_FOO and PICTOR_foo both map to the field name 'foo',
// so pictor_reflect rejects the redeclaration. Never add this to pictor_add_shader;
// the ctest target invokes the reflector directly and expects the diagnostic.

#ifndef PICTOR_FOO
#define PICTOR_FOO 1
#endif

#ifndef PICTOR_foo
#define PICTOR_foo 2
#endif

[shader("compute")]
[numthreads(1, 1, 1)]
void bad_collision_compute(uint3 tid : SV_DispatchThreadID)
{
    ((uint*)0)[tid.x] = PICTOR_FOO + PICTOR_foo;
}
