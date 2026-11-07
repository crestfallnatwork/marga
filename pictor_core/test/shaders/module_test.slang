/* SPDX-License-Identifier: MPL-2.0 */

import wave_ops;

struct ModuleParams {
    uint64_t out_addr;
    uint32_t count;
};

[[vk::push_constant]]
uniform ModuleParams p;

[shader("compute")]
[numthreads(32, 1, 1)]
void module_compute(uint3 tid : SV_DispatchThreadID)
{
    ((uint*)p.out_addr)[tid.x] = wave_mix(tid.x) + wave_broadcast(p.count);
}
