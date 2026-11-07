/* SPDX-License-Identifier: MPL-2.0 */

#include "pictor_core/context/context.hpp"

#include "tuner_test.shader.h"

#include <cstdint>
#include <cstdio>
#include <type_traits>
#include <vector>

using namespace pictor;

// Every trait of a shader that declares tunables aliases the one generated struct.
static_assert(std::is_same_v<tuner_test_shader::TunerCompute::Tunables,
                             tuner_test_shader::TunerTestTunables>);
static_assert(
    std::is_same_v<tuner_test_shader::TunerMesh::Tunables, tuner_test_shader::TunerTestTunables>);

// `create(ctx, nullptr)` must keep resolving to the untuned overload. The tuned overload is
// constrained on the second argument being invocable as a tuner, so a nullptr is not a
// candidate; if that constraint regressed, this call would be ambiguous and fail to compile.
static_assert(requires(Context& ctx) {
    tuner_test_shader::TunerComputeExecutor::create(ctx, nullptr);
});
static_assert(requires(Context& ctx) { tuner_test_shader::TUNERExecutor::create(ctx, nullptr); });

// A code type without a `Tunables` alias stays compile-clean: the tuned overload is not viable
// for it, so `create(ctx, nullptr)` still selects the untuned overload.
struct PlainCode {};
struct PlainParams {
    uint32_t value;
};

static_assert(requires(Context& ctx) {
    ComputeExecutor<PlainCode, PlainParams>::create(ctx, nullptr);
});

// Dependent so an ill-formed call under substitution is false rather than an error. A callable
// cannot be injected for a code type without tunables, even one whose shape matches a tuner.
template <typename TTuner>
concept TunedPlain = requires(Context& ctx, TTuner&& tuner) {
    ComputeExecutor<PlainCode, PlainParams>::create(ctx, tuner);
};

static_assert(!TunedPlain<void (*)(const DeviceInfo&, PlainCode&)>);
static_assert(!TunedPlain<void (*)(const DeviceInfo&, tuner_test_shader::TunerTestTunables&)>);

namespace {

template <typename TExecutor>
std::vector<uint32_t> run(Context& ctx, TExecutor& exec, const AllocationInfo& d_out,
                          AllocationInfo& h_out, uint32_t count)
{
    tuner_test_shader::TunerParams params{};
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

bool expect_values(const std::vector<uint32_t>& got, uint32_t bias, const char* label)
{
    for (uint32_t i = 0; i < got.size(); ++i) {
        const uint32_t want = i + bias;
        if (got[i] != want) {
            std::fprintf(stderr, "tuner_test: %s mismatch at %u: got %u want %u\n", label, i,
                         got[i], want);
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
        std::fprintf(stderr, "tuner_test: Context::create failed (vulkan %d)\n",
                     static_cast<int>(ctx_res.err()));
        return 1;
    }
    auto ctx = std::move(ctx_res.val());

    bool ok = true;

    // The generated defaults are the shader's declared values.
    tuner_test_shader::TunerTestTunables declared;
    if (declared.threads != 32 || declared.bias != 1) {
        std::fprintf(stderr, "tuner_test: declared defaults mismatch (threads=%u bias=%u)\n",
                     declared.threads, declared.bias);
        return 1;
    }

    const DeviceInfo* tuned_with = nullptr;
    auto tuner                   = [&tuned_with](const DeviceInfo& device,
                                                 tuner_test_shader::TunerTestTunables& tunables) {
        tuned_with       = &device;
        tunables.threads = 16;
        tunables.bias    = 100;
    };

    auto untuned_res = tuner_test_shader::TunerComputeExecutor::create(*ctx);
    if (!untuned_res.ok()) {
        std::fprintf(stderr, "tuner_test: untuned compute create failed (vulkan %d)\n",
                     static_cast<int>(untuned_res.err()));
        return 1;
    }
    auto untuned = std::move(untuned_res.val());

    auto tuned_res = tuner_test_shader::TunerComputeExecutor::create(*ctx, tuner);
    if (!tuned_res.ok()) {
        std::fprintf(stderr, "tuner_test: tuned compute create failed (vulkan %d)\n",
                     static_cast<int>(tuned_res.err()));
        return 1;
    }
    auto tuned = std::move(tuned_res.val());

    auto untuned_mesh_res = tuner_test_shader::TUNERExecutor::create(*ctx);
    if (!untuned_mesh_res.ok()) {
        std::fprintf(stderr, "tuner_test: untuned mesh create failed (vulkan %d)\n",
                     static_cast<int>(untuned_mesh_res.err()));
        return 1;
    }
    auto untuned_mesh = std::move(untuned_mesh_res.val());

    auto tuned_mesh_res = tuner_test_shader::TUNERExecutor::create(*ctx, tuner);
    if (!tuned_mesh_res.ok()) {
        std::fprintf(stderr, "tuner_test: tuned mesh create failed (vulkan %d)\n",
                     static_cast<int>(tuned_mesh_res.err()));
        return 1;
    }
    auto tuned_mesh = std::move(tuned_mesh_res.val());

    if (tuned_with != &ctx->device_info) {
        std::fprintf(stderr, "tuner_test: tuner did not receive ctx.device_info\n");
        ok = false;
    }

    // A [numthreads]-participating tunable must show up in the reflected workgroup size.
    if (untuned->wgX != 32 || tuned->wgX != 16) {
        std::fprintf(stderr, "tuner_test: compute wgX untuned=%u tuned=%u (want 32/16)\n",
                     untuned->wgX, tuned->wgX);
        ok = false;
    }
    if (untuned_mesh->taskWgX != 32 || untuned_mesh->meshWgX != 32 || tuned_mesh->taskWgX != 16 ||
        tuned_mesh->meshWgX != 16) {
        std::fprintf(stderr, "tuner_test: mesh sizes untuned=%u/%u tuned=%u/%u (want 32/16)\n",
                     untuned_mesh->taskWgX, untuned_mesh->meshWgX, tuned_mesh->taskWgX,
                     tuned_mesh->meshWgX);
        ok = false;
    }

    constexpr uint32_t kCount = 64;
    auto d_out = ctx->pool_allocator->allocate(kCount * sizeof(uint32_t), alignof(uint32_t));
    auto h_out = ctx->pinned_allocator->allocate(kCount * sizeof(uint32_t), alignof(uint32_t));

    const std::vector<uint32_t> got_untuned  = run(*ctx, *untuned, d_out, h_out, kCount);
    const std::vector<uint32_t> got_tuned    = run(*ctx, *tuned, d_out, h_out, kCount);
    const std::vector<uint32_t> got_mesh     = run(*ctx, *untuned_mesh, d_out, h_out, kCount);
    const std::vector<uint32_t> got_mesh_tun = run(*ctx, *tuned_mesh, d_out, h_out, kCount);

    ok = expect_values(got_untuned, 1, "untuned compute") && ok;
    ok = expect_values(got_tuned, 100, "tuned compute") && ok;
    ok = expect_values(got_mesh, 1, "untuned mesh") && ok;
    ok = expect_values(got_mesh_tun, 100, "tuned mesh") && ok;

    ctx->deallocate(d_out);
    ctx->deallocate(h_out);

    std::printf("%s\n",
                ok ? "injected shader tuner test passed" : "INJECTED SHADER TUNER TEST FAILED");
    return ok ? 0 : 1;
}
