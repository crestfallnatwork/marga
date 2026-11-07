/* SPDX-License-Identifier: MPL-2.0 */

import "imported_push_block";

[shader("compute")]
[numthreads(1, 1, 1)]
void import_compute(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x == 0)
        write_imported();
}
