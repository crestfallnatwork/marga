/* SPDX-License-Identifier: MPL-2.0 */

// Imported by name through INCLUDE_DIRS: pictor_add_shader(... INCLUDE_DIRS ops)
// makes ops/ a search path, so `import wave_ops;` finds ops/wave_ops.slang.
import wave_ops;

struct MixParams {
    uint64_t out_addr;
    uint32_t count;
};

[[vk::push_constant]]
uniform MixParams p;

// WAVE_SIZE is the device subgroup size, injected when the executor compiles this
// source at runtime. One workgroup of one wave keeps the wave_broadcast lane simple.
[shader("compute")]
[numthreads(WAVE_SIZE, 1, 1)]
void mix_compute(uint3 tid : SV_DispatchThreadID)
{
    ((uint*)p.out_addr)[tid.x] = wave_mix(tid.x) + wave_broadcast(p.count);
}
