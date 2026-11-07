/* SPDX-License-Identifier: MPL-2.0 */

// Two blocks with the same type are one layout. Every entry binds SharedIn, including the one
// that references both variables; the block-type resolution must not reject it.

struct SharedIn {
    uint32_t* out;
    uint32_t  value;
};

[[vk::push_constant]]
uniform SharedIn first;

[[vk::push_constant]]
uniform SharedIn second;

[shader("compute")]
[numthreads(1, 1, 1)]
void first_compute(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x == 0 && tid.y == 0 && tid.z == 0)
        *first.out = first.value;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void second_compute(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x == 0 && tid.y == 0 && tid.z == 0)
        *second.out = second.value;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void both_compute(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x == 0 && tid.y == 0 && tid.z == 0)
        *first.out = first.value + second.value;
}
