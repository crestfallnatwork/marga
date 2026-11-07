/* SPDX-License-Identifier: MPL-2.0 */

#include "pictor_core/context/context.hpp"
#include "pictor_core/execution/shader_compiler.hpp"
#include "pictor_core/execution/shader_defines.hpp"

#include "tunable_test.shader.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

using namespace pictor;

static_assert(!std::is_copy_constructible_v<ShaderDefines>);
static_assert(!std::is_copy_assignable_v<ShaderDefines>);
static_assert(std::is_move_constructible_v<ShaderDefines>);
static_assert(std::is_move_assignable_v<ShaderDefines>);

namespace {

bool hasDefine(std::span<const ShaderDefine> defines, std::string_view name, std::string_view value)
{
    return std::ranges::any_of(defines, [&](const ShaderDefine& define) {
        return define.name == name && define.value == value;
    });
}

template <typename TExecutor>
std::vector<uint32_t> run(Context& ctx, TExecutor& exec, const AllocationInfo& d_out,
                          AllocationInfo& h_out, uint32_t count)
{
    tunable_test_shader::TunableParams params{};
    params.out_addr = d_out.address;
    params.count    = count;

    exec.record(count, 1, 1, params);
    ctx.barrier(d_out, Access::Write, Access::TransferRead);
    ctx.copy(d_out, h_out);
    ctx.fsync();
    ctx.invalidate(h_out);

    const auto* src = static_cast<const uint32_t*>(h_out.mappedPtr);
    return {src, src + count};
}

} // namespace

int main()
{
    auto ctx_res = Context::create();
    if (!ctx_res.ok()) {
        std::fprintf(stderr, "tunable_test: Context::create failed (vulkan %d)\n",
                     static_cast<int>(ctx_res.err()));
        return 1;
    }
    auto ctx = std::move(ctx_res.val());

    // The generated struct's defaults must mirror the shader's live guarded defaults, including
    // the tunable declared in the imported module. The commented-out, #if 0 and false #ifdef
    // declarations in the shader must not appear: a leaked ghost would change defines().size().
    tunable_test_shader::TunableTestTunables tunables;
    if (tunables.scale != 3 || tunables.offset != 1 || tunables.enable_extra ||
        tunables.module_bias != 10 || tunables.octal != 8) {
        std::fprintf(stderr,
                     "tunable_test: generated defaults mismatch (scale=%u offset=%u "
                     "enable_extra=%d module_bias=%u octal=%u)\n",
                     tunables.scale, tunables.offset, static_cast<int>(tunables.enable_extra),
                     tunables.module_bias, tunables.octal);
        return 1;
    }

    // defines() round-trips those defaults as preprocessor values. Leading-zero defaults are
    // normalized to decimal, matching the value Slang compiles (010 is octal 8).
    ShaderDefines roundTrip = tunables.defines();
    auto roundTripView      = roundTrip.view();
    if (roundTripView.size() != 5 || !hasDefine(roundTripView, "PICTOR_SCALE", "3") ||
        !hasDefine(roundTripView, "PICTOR_OFFSET", "1") ||
        !hasDefine(roundTripView, "PICTOR_ENABLE_EXTRA", "0") ||
        !hasDefine(roundTripView, "PICTOR_MODULE_BIAS", "10") ||
        !hasDefine(roundTripView, "PICTOR_OCTAL", "8")) {
        std::fprintf(stderr, "tunable_test: defines() did not round-trip the defaults\n");
        return 1;
    }

    constexpr uint32_t kCount = 64;

    auto default_res = tunable_test_shader::TunableComputeExecutor::create(*ctx);
    if (!default_res.ok()) {
        std::fprintf(stderr, "tunable_test: default executor creation failed (vulkan %d)\n",
                     static_cast<int>(default_res.err()));
        return 1;
    }
    auto default_exec = std::move(default_res.val());

    ShaderDefines overrides;
    overrides.add("PICTOR_SCALE", "5");
    overrides.add("PICTOR_OFFSET", "2");
    overrides.add("PICTOR_ENABLE_EXTRA", "true");

    auto override_res = tunable_test_shader::TunableComputeExecutor::create(
        *ctx, nullptr, CompileOptions{overrides.view()});
    if (!override_res.ok()) {
        std::fprintf(stderr, "tunable_test: override executor creation failed (vulkan %d)\n",
                     static_cast<int>(override_res.err()));
        return 1;
    }
    auto override_exec = std::move(override_res.val());

    // MeshExecutor must forward CompileOptions to both the task and mesh stage compiles.
    auto default_mesh_res = tunable_test_shader::TUNABLEExecutor::create(*ctx);
    if (!default_mesh_res.ok()) {
        std::fprintf(stderr, "tunable_test: default mesh executor creation failed (vulkan %d)\n",
                     static_cast<int>(default_mesh_res.err()));
        return 1;
    }
    auto default_mesh = std::move(default_mesh_res.val());

    auto override_mesh_res = tunable_test_shader::TUNABLEExecutor::create(
        *ctx, nullptr, CompileOptions{overrides.view()});
    if (!override_mesh_res.ok()) {
        std::fprintf(stderr, "tunable_test: override mesh executor creation failed (vulkan %d)\n",
                     static_cast<int>(override_mesh_res.err()));
        return 1;
    }
    auto override_mesh = std::move(override_mesh_res.val());

    auto d_out = ctx->pool_allocator->allocate(kCount * sizeof(uint32_t), alignof(uint32_t));
    auto h_out = ctx->pinned_allocator->allocate(kCount * sizeof(uint32_t), alignof(uint32_t));

    bool ok = true;

    const std::vector<uint32_t> got_default  = run(*ctx, *default_exec, d_out, h_out, kCount);
    const std::vector<uint32_t> got_override = run(*ctx, *override_exec, d_out, h_out, kCount);

    for (uint32_t i = 0; i < kCount; ++i) {
        const uint32_t want_default  = (i * 3u) + 1u + 8u + 10u;
        const uint32_t want_override = (i * 5u) + 2u + 100u + 8u + 10u;
        if (got_default[i] != want_default) {
            std::fprintf(stderr, "tunable_test: default mismatch at %u: got %u want %u\n", i,
                         got_default[i], want_default);
            ok = false;
            break;
        }
        if (got_override[i] != want_override) {
            std::fprintf(stderr, "tunable_test: override mismatch at %u: got %u want %u\n", i,
                         got_override[i], want_override);
            ok = false;
            break;
        }
        if (got_default[i] == got_override[i]) {
            std::fprintf(stderr, "tunable_test: defines did not reach Slang at %u (both %u)\n", i,
                         got_default[i]);
            ok = false;
            break;
        }
    }

    const std::vector<uint32_t> mesh_default  = run(*ctx, *default_mesh, d_out, h_out, kCount);
    const std::vector<uint32_t> mesh_override = run(*ctx, *override_mesh, d_out, h_out, kCount);

    for (uint32_t i = 0; i < kCount; ++i) {
        const uint32_t want_default  = (i * 3u) + 1u;
        const uint32_t want_override = (i * 5u) + 2u;
        if (mesh_default[i] != want_default) {
            std::fprintf(stderr, "tunable_test: mesh default mismatch at %u: got %u want %u\n", i,
                         mesh_default[i], want_default);
            ok = false;
            break;
        }
        if (mesh_override[i] != want_override) {
            std::fprintf(stderr, "tunable_test: mesh override mismatch at %u: got %u want %u\n", i,
                         mesh_override[i], want_override);
            ok = false;
            break;
        }
        if (mesh_default[i] == mesh_override[i]) {
            std::fprintf(stderr, "tunable_test: mesh defines did not reach Slang at %u (both %u)\n",
                         i, mesh_default[i]);
            ok = false;
            break;
        }
    }

    // Both compile overloads must accept the options. The bare-source overload compiles a
    // self-contained snippet twice with different values and checks the SPIR-V differs, so
    // the defines are proven to reach Slang on that path too.
    const std::string inlineSource = R"(
#ifndef PICTOR_INLINE_VALUE
#define PICTOR_INLINE_VALUE 7
#endif

struct InlineParams {
    uint64_t out_addr;
};

[[vk::push_constant]]
uniform InlineParams ip;

[shader("compute")]
[numthreads(1, 1, 1)]
void inline_compute(uint3 tid : SV_DispatchThreadID)
{
    ((uint*)ip.out_addr)[tid.x] = PICTOR_INLINE_VALUE;
}
)";

    ShaderDefines inlineA;
    inlineA.add("PICTOR_INLINE_VALUE", "7");
    ShaderDefines inlineB;
    inlineB.add("PICTOR_INLINE_VALUE", "9");

    auto bareA = ctx->shader_compiler->compile(inlineSource, "inline_compute", ShaderStage::Compute,
                                               ctx->subgroup_size, CompileOptions{inlineA.view()});
    auto bareB = ctx->shader_compiler->compile(inlineSource, "inline_compute", ShaderStage::Compute,
                                               ctx->subgroup_size, CompileOptions{inlineB.view()});
    if (!bareA.success || !bareB.success) {
        std::fprintf(stderr, "tunable_test: bare-source compile failed: %s\n",
                     bareA.success ? bareB.diagnostics.c_str() : bareA.diagnostics.c_str());
        ok = false;
    } else if (bareA.spirv == bareB.spirv) {
        std::fprintf(stderr, "tunable_test: bare-source defines did not reach Slang\n");
        ok = false;
    }

    // WAVE_SIZE is a device fact injected by the runtime; a caller-supplied one must be
    // rejected with a diagnostic instead of being silently ignored.
    const std::array waveDefine = {ShaderDefine{"WAVE_SIZE", "64"}};
    auto waveRes =
        ctx->shader_compiler->compile(inlineSource, "inline_compute", ShaderStage::Compute,
                                      ctx->subgroup_size, CompileOptions{waveDefine});
    if (waveRes.success || waveRes.diagnostics.find("WAVE_SIZE") == std::string::npos) {
        std::fprintf(stderr, "tunable_test: WAVE_SIZE define was not rejected: %s\n",
                     waveRes.success ? "compile succeeded" : waveRes.diagnostics.c_str());
        ok = false;
    }

    auto viaSources = ctx->shader_compiler->compile(
        tunable_test_shader_sources::cSet, "tunable_compute", ShaderStage::Compute,
        ctx->subgroup_size, CompileOptions{overrides.view()});
    if (!viaSources.success) {
        std::fprintf(stderr, "tunable_test: ShaderSources compile failed: %s\n",
                     viaSources.diagnostics.c_str());
        ok = false;
    }

    ctx->deallocate(d_out);
    ctx->deallocate(h_out);

    std::printf("%s\n", ok ? "shader tunables test passed" : "SHADER TUNABLES TEST FAILED");
    return ok ? 0 : 1;
}
