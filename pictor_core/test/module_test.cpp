/* SPDX-License-Identifier: MPL-2.0 */

#include "pictor_core/context/context.hpp"

#include "module_test.shader.h"

#include <cstdint>
#include <cstdio>

using namespace pictor;

int main()
{
    auto ctx_res = Context::create();
    if (!ctx_res.ok()) {
        std::fprintf(stderr, "failed to create pictor context (vulkan %d)\n",
                     static_cast<int>(ctx_res.err()));
        return 1;
    }
    auto ctx = std::move(ctx_res.val());

    auto exec_res = module_test_shader::ModuleComputeExecutor::create(*ctx);
    if (!exec_res.ok()) {
        std::fprintf(stderr, "failed to create the module executor (vulkan %d)\n",
                     static_cast<int>(exec_res.err()));
        return 1;
    }
    auto exec = std::move(exec_res.val());

    constexpr uint32_t kCount  = 32;
    constexpr uint32_t kCount0 = 7;
    constexpr size_t kBytes    = kCount * sizeof(uint32_t);

    auto d_out = ctx->pool_allocator->allocate(kBytes, alignof(uint32_t));
    auto h_out = ctx->pinned_allocator->allocate(kBytes, alignof(uint32_t));

    module_test_shader::ModuleParams params{};
    params.out_addr = d_out.address;
    params.count    = kCount0;

    exec->record(kCount, 1, 1, params);
    ctx->barrier(d_out, Access::Write, Access::TransferRead);
    ctx->copy(d_out, h_out);
    ctx->fsync();
    ctx->invalidate(h_out);

    const auto* got = static_cast<const uint32_t*>(h_out.mappedPtr);
    bool ok         = true;
    for (uint32_t i = 0; i < kCount; ++i) {
        const uint32_t want = (i * 3u) + 1u + kCount0;
        if (got[i] != want) {
            std::fprintf(stderr, "mismatch at %u: got %u want %u\n", i, got[i], want);
            ok = false;
            break;
        }
    }

    ctx->deallocate(d_out);
    ctx->deallocate(h_out);

    std::printf("%s\n", ok ? "shader module test passed" : "SHADER MODULE TEST FAILED");
    return ok ? 0 : 1;
}
