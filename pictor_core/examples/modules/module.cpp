/* SPDX-License-Identifier: MPL-2.0 */

// Device-side module example. See docs/shader-codegen.md for the full pipeline guide.
//
// mix.slang imports wave_ops (via INCLUDE_DIRS), which imports scale; the generated
// "mix.shader.h" embeds all three sources and the runtime compiles them from memory,
// so nothing has to ship alongside the binary.

#include "pictor_core/context/context.hpp"

#include "mix.shader.h"

#include <cstdint>
#include <cstdio>

using namespace pictor;

int main()
{
    auto ctx_res = Context::create();
    if (!ctx_res.ok()) {
        std::fprintf(stderr, "example_modules: Context::create failed (vulkan %d)\n",
                     static_cast<int>(ctx_res.err()));
        return 1;
    }
    auto ctx = std::move(ctx_res.val());

    auto exec_res = mix_shader::MixComputeExecutor::create(*ctx);
    if (!exec_res.ok()) {
        std::fprintf(stderr, "example_modules: executor creation failed (vulkan %d)\n",
                     static_cast<int>(exec_res.err()));
        return 1;
    }
    auto exec = std::move(exec_res.val());

    // mix.slang declares [numthreads(WAVE_SIZE, 1, 1)] and WAVE_SIZE is the device
    // subgroup size, so one workgroup covers exactly one wave.
    const uint32_t count = ctx->subgroup_size;
    const size_t bytes   = count * sizeof(uint32_t);

    auto d_out = ctx->pool_allocator->allocate(bytes, alignof(uint32_t));
    auto h_out = ctx->pinned_allocator->allocate(bytes, alignof(uint32_t));

    mix_shader::MixParams params{};
    params.out_addr = d_out.address;
    params.count    = count;

    exec->record(count, 1, 1, params);

    ctx->barrier(d_out, Access::Write, Access::TransferRead);
    ctx->copy(d_out, h_out);
    ctx->fsync();
    ctx->invalidate(h_out);

    bool ok         = true;
    const auto* got = static_cast<const uint32_t*>(h_out.mappedPtr);
    for (uint32_t i = 0; i < count; ++i) {
        // wave_mix(i) = i * 5 + 3 (through scale_value), wave_broadcast(count) = count.
        const uint32_t want = (i * 5u) + 3u + count;
        if (got[i] != want) {
            std::fprintf(stderr, "example_modules: mismatch at %u: got %u want %u\n", i, got[i],
                         want);
            ok = false;
            break;
        }
    }

    ctx->deallocate(d_out);
    ctx->deallocate(h_out);

    std::printf("example_modules: %s\n", ok ? "PASS" : "FAILED");
    return ok ? 0 : 1;
}
