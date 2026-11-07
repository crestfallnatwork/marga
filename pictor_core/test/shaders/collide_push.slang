/* SPDX-License-Identifier: MPL-2.0 */

import "collide_push_block";

struct LocalIn {
    uint32_t* out;
};

[[vk::push_constant]]
uniform LocalIn shared;

[shader("compute")]
[numthreads(1, 1, 1)]
void collide_compute(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x == 0)
        write_shared();
}
