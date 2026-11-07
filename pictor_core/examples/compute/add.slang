/* SPDX-License-Identifier: MPL-2.0 */

// Minimal compute shader: c[i] = a[i] + b[i], with the three buffers passed as raw
// buffer-device addresses (BDA) in the push-constant block.
//
// pictor_add_shader() runs pictor_reflect at build time and emits add.slang.h (the
// embedded source closure) and add.shader.h (the C++ interface used below). The
// shader source itself needs no annotations beyond normal Slang.

struct AddParams {
    uint64_t a_addr;
    uint64_t b_addr;
    uint64_t out_addr;
    uint32_t count;
};

[[vk::push_constant]]
uniform AddParams p;

// Entry points are grouped by the prefix before the last underscore: add_compute
// becomes add_shader::AddCompute (trait) and add_shader::AddComputeExecutor (alias).
[shader("compute")]
[numthreads(64, 1, 1)]
void add_compute(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x < p.count)
        ((float*)p.out_addr)[tid.x] = ((float*)p.a_addr)[tid.x] + ((float*)p.b_addr)[tid.x];
}
