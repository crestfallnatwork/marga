/* SPDX-License-Identifier: MPL-2.0 */

// One entry may use only one push-constant layout; this one references two different blocks.

struct FirstIn {
    uint32_t* out;
};

struct SecondIn {
    uint32_t value;
};

[[vk::push_constant]]
uniform FirstIn first;

[[vk::push_constant]]
uniform SecondIn second;

[shader("compute")]
[numthreads(1, 1, 1)]
void two_blocks_compute(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x == 0 && tid.y == 0 && tid.z == 0)
        *first.out = second.value;
}
