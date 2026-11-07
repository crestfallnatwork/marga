/* SPDX-License-Identifier: MPL-2.0 */

// Minimal compute example. See docs/shader-codegen.md for the full pipeline guide.
//
// The build generates "add.shader.h" (via pictor_add_shader in CMakeLists.txt) which
// declares add_shader::AddParams and add_shader::AddComputeExecutor.

#include "pictor_core/context/context.hpp"
#include "pictor_core/profiling/gputrace.hpp"

#include "add.shader.h"

#include <cstdint>
#include <cstdio>

using namespace pictor;

int main()
{
    auto ctx_res = Context::create();
    if (!ctx_res.ok()) {
        std::fprintf(stderr, "example_compute: Context::create failed (vulkan %d)\n",
                     static_cast<int>(ctx_res.err()));
        return 1;
    }
    auto ctx = std::move(ctx_res.val());

    auto exec_res = add_shader::AddComputeExecutor::create(*ctx);
    if (!exec_res.ok()) {
        std::fprintf(stderr, "example_compute: executor creation failed (vulkan %d)\n",
                     static_cast<int>(exec_res.err()));
        return 1;
    }
    auto exec = std::move(exec_res.val());

    constexpr uint32_t kCount = 1024;
    constexpr size_t kBytes   = kCount * sizeof(float);

    auto d_a   = ctx->pool_allocator->allocate(kBytes, alignof(float));
    auto d_b   = ctx->pool_allocator->allocate(kBytes, alignof(float));
    auto d_out = ctx->pool_allocator->allocate(kBytes, alignof(float));
    auto h_a   = ctx->pinned_allocator->allocate(kBytes, alignof(float));
    auto h_b   = ctx->pinned_allocator->allocate(kBytes, alignof(float));
    auto h_out = ctx->pinned_allocator->allocate(kBytes, alignof(float));

    auto* a = static_cast<float*>(h_a.mappedPtr);
    auto* b = static_cast<float*>(h_b.mappedPtr);
    for (uint32_t i = 0; i < kCount; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(i) * 0.5f;
    }
    ctx->flush(h_a);
    ctx->flush(h_b);

    ctx->copy(h_a, d_a);
    ctx->copy(h_b, d_b);
    ctx->barrier(d_a, Access::TransferWrite, Access::Read);
    ctx->barrier(d_b, Access::TransferWrite, Access::Read);

    add_shader::AddParams params{};
    params.a_addr   = d_a.address;
    params.b_addr   = d_b.address;
    params.out_addr = d_out.address;
    params.count    = kCount;

    // record(x, y, z, params) dispatches ceil(x / workgroup_size) workgroups. The
    // 64x1x1 workgroup size declared in add.slang was reflected at executor creation;
    // the host never repeats it.
    //
    // ScopedRange is a no-op unless the app is run under GPU Trace with PICTOR_NGFX_SDK=1
    // (see docs/profiling/gputrace.md); then it starts/stops an app-driven trace around this pass.
    {
        gputrace::ScopedRange trace_range;
        exec->record(kCount, 1, 1, params);

        ctx->barrier(d_out, Access::Write, Access::TransferRead);
        ctx->copy(d_out, h_out);
        ctx->fsync();
    }
    ctx->invalidate(h_out);

    bool ok   = true;
    auto* out = static_cast<float*>(h_out.mappedPtr);
    for (uint32_t i = 0; i < kCount; ++i) {
        const float want = a[i] + b[i];
        if (out[i] != want) {
            std::fprintf(stderr, "example_compute: mismatch at %u: got %f want %f\n", i, out[i],
                         want);
            ok = false;
            break;
        }
    }

    ctx->deallocate(d_a);
    ctx->deallocate(d_b);
    ctx->deallocate(d_out);
    ctx->deallocate(h_a);
    ctx->deallocate(h_b);
    ctx->deallocate(h_out);

    std::printf("example_compute: %s\n", ok ? "PASS" : "FAILED");
    return ok ? 0 : 1;
}
