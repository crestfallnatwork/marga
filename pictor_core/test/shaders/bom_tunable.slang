#ifndef PICTOR_BOM
#define PICTOR_BOM 5
#endif

/* SPDX-License-Identifier: MPL-2.0 */

// S5 regression fixture: the leading UTF-8 BOM must not hide the first declaration.
// Never add this to pictor_add_shader; the ctest target runs the reflector and
// inspects --dump.

[shader("compute")]
[numthreads(1, 1, 1)]
void bom_compute(uint3 tid : SV_DispatchThreadID)
{
    ((uint*)0)[tid.x] = PICTOR_BOM;
}
