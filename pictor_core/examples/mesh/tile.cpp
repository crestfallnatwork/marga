/* SPDX-License-Identifier: MPL-2.0 */

// Task + mesh example. See docs/shader-codegen.md for the full pipeline guide.
//
// The build generates "tile.shader.h" which declares tile_shader::TileParams and
// tile_shader::TILEExecutor.

#include "pictor_core/context/context.hpp"

#include "tile.shader.h"

#include <cstdint>
#include <cstdio>

using namespace pictor;

namespace {

bool verify(Context& ctx, const AllocationInfo& d_out, AllocationInfo& h_out, uint32_t count,
            const char* label)
{
    ctx.barrier(d_out, Access::Write, Access::TransferRead);
    ctx.copy(d_out, h_out);
    ctx.fsync();
    ctx.invalidate(h_out);

    const auto* got = static_cast<const uint32_t*>(h_out.mappedPtr);
    for (uint32_t i = 0; i < count; ++i) {
        if (got[i] != i) {
            std::fprintf(stderr, "example_mesh: %s mismatch at %u: got %u want %u\n", label, i,
                         got[i], i);
            return false;
        }
    }
    return true;
}

} // namespace

int main()
{
    auto ctx_res = Context::create();
    if (!ctx_res.ok()) {
        std::fprintf(stderr, "example_mesh: Context::create failed (vulkan %d)\n",
                     static_cast<int>(ctx_res.err()));
        return 1;
    }
    auto ctx = std::move(ctx_res.val());

    auto exec_res = tile_shader::TILEExecutor::create(*ctx);
    if (!exec_res.ok()) {
        std::fprintf(stderr, "example_mesh: executor creation failed (vulkan %d)\n",
                     static_cast<int>(exec_res.err()));
        return 1;
    }
    auto exec = std::move(exec_res.val());

    // 200 elements is not a multiple of the 64-thread mesh workgroup, so both dispatch
    // paths leave some lanes idle and the shader's bounds check is exercised.
    constexpr uint32_t kCount = 200;
    constexpr size_t kBytes   = kCount * sizeof(uint32_t);

    auto d_out_a = ctx->pool_allocator->allocate(kBytes, alignof(uint32_t));
    auto d_out_b = ctx->pool_allocator->allocate(kBytes, alignof(uint32_t));
    auto h_out_a = ctx->pinned_allocator->allocate(kBytes, alignof(uint32_t));
    auto h_out_b = ctx->pinned_allocator->allocate(kBytes, alignof(uint32_t));

    tile_shader::TileParams params{};
    params.count = kCount;

    // record(x, y, z, params) divides x by the *task* workgroup size (32 here) and
    // dispatches ceil(200 / 32) = 7 task groups; each dispatches one mesh group.
    params.out_addr = d_out_a.address;
    exec->record(kCount, 1, 1, params);

    // record_raw(x, y, z, params) passes the dispatch counts through unchanged, so the
    // caller divides by the mesh workgroup size; meshWgX is a public executor constant.
    const uint32_t groups = (kCount + exec->meshWgX - 1) / exec->meshWgX;
    params.out_addr       = d_out_b.address;
    exec->record_raw(groups, 1, 1, params);

    bool ok = verify(*ctx, d_out_a, h_out_a, kCount, "record");
    ok &= verify(*ctx, d_out_b, h_out_b, kCount, "record_raw");

    ctx->deallocate(d_out_a);
    ctx->deallocate(d_out_b);
    ctx->deallocate(h_out_a);
    ctx->deallocate(h_out_b);

    std::printf("example_mesh: %s\n", ok ? "PASS" : "FAILED");
    return ok ? 0 : 1;
}
