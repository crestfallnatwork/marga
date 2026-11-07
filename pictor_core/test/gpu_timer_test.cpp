/* SPDX-License-Identifier: MPL-2.0 */

#include "pictor_core/context/context.hpp"
#include "pictor_core/execution/gpu_timer.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <span>
#include <string_view>
#include <utility>

using namespace pictor;

namespace {

bool g_ok = true;

void check(bool condition, const char* what)
{
    if (!condition) {
        std::fprintf(stderr, "gpu timer check failed: %s\n", what);
        g_ok = false;
    }
}

bool sane(const GpuTimer::LapResult& lap)
{
    return std::isfinite(lap.ms) && lap.ms >= 0.0;
}

bool not_ready(const Expected<std::span<const GpuTimer::LapResult>, vk::Result>& result)
{
    return !result.ok() && result.err() == vk::Result::eNotReady;
}

bool names_match(std::span<const GpuTimer::LapResult> laps,
                 std::initializer_list<std::string_view> names)
{
    if (laps.size() != names.size()) {
        return false;
    }
    size_t i = 0;
    for (std::string_view name : names) {
        if (laps[i++].name != name) {
            return false;
        }
    }
    return true;
}

void print_laps(std::span<const GpuTimer::LapResult> laps)
{
    for (const auto& lap : laps) {
        std::printf("  %.*s: %.4f ms\n", static_cast<int>(lap.name.size()), lap.name.data(),
                    lap.ms);
    }
}

} // namespace

int main()
{
    auto ctx_res = Context::create();
    if (!ctx_res.ok()) {
        std::fprintf(stderr, "failed to create pictor context (vulkan %d)\n",
                     static_cast<int>(ctx_res.err()));
        return 1;
    }
    auto ctx = std::move(ctx_res.val());

    constexpr size_t kBytes = 64u << 20u;
    auto d_a                = ctx->pool_allocator->allocate(kBytes, 16);
    auto d_b                = ctx->pool_allocator->allocate(kBytes, 16);
    if (!d_a.buffer || !d_b.buffer) {
        std::fprintf(stderr, "buffer allocation failed\n");
        return 1;
    }

    {
        GpuTimer disabled;
        check(!disabled.enabled(), "default timer is disabled");
        auto inactive = disabled.lap("off");
        check(!inactive.active(), "disabled lap is inactive");
        auto empty = disabled.read();
        check(empty.ok() && empty->empty(), "disabled read is an empty success");
        check(disabled.lapCount() == 0, "disabled lapCount is zero");
        disabled.reset();
        check(!disabled.enabled(), "reset keeps a disabled timer disabled");
    }

    {
        auto disabled_res = GpuTimer::create(*ctx, 0);
        check(disabled_res.ok(), "create(ctx, 0) succeeds");
        if (disabled_res.ok()) {
            GpuTimer& disabled = *disabled_res;
            check(!disabled.enabled(), "create(ctx, 0) is disabled");
            auto inactive = disabled.lap("off");
            check(!inactive.active(), "created disabled lap is inactive");
            auto empty = disabled.read();
            check(empty.ok() && empty->empty(), "created disabled read is an empty success");
        }
    }

    {
        auto invalid = GpuTimer::create(*ctx, 2, StreamHandle{999});
        check(!invalid.ok() && invalid.err() == vk::Result::eErrorInvalidExternalHandle,
              "an invalid stream handle is rejected");
    }

    auto timer_res = GpuTimer::create(*ctx, 3);
    if (!timer_res.ok()) {
        check(timer_res.err() == vk::Result::eErrorFeatureNotPresent,
              "the only expected create failure is a missing timestamp feature");
        std::printf("gpu_timer: skipped enabled tests (timestamps unsupported)\n");
        ctx->deallocate(d_a);
        ctx->deallocate(d_b);
        return g_ok ? 0 : 1;
    }

    GpuTimer timer = std::move(timer_res.val());
    check(timer.enabled(), "create(ctx, 3) is enabled");
    check(timer.lapCount() == 0, "a fresh timer has no laps");

    {
        auto empty = timer.read();
        check(empty.ok() && empty->empty(), "reading an empty session is an empty success");
    }

    {
        auto lap = timer.lap("pending");
        check(lap.active(), "pending lap is active");
        ctx->fill(d_a, 0u);
    }

    auto before_submit = timer.read();
    check(not_ready(before_submit), "read before submit is eNotReady");
    check(ctx->submit() == vk::Result::eSuccess, "submit before fsync succeeds");

    auto after_submit = timer.read();
    check(not_ready(after_submit), "read after submit but before fsync is eNotReady");

    check(ctx->fsync() == vk::Result::eSuccess, "fsync after the pending session succeeds");
    auto pending = timer.read();
    check(pending.ok(), "reading after completion succeeds");
    if (pending.ok()) {
        print_laps(*pending);
        check(names_match(*pending, {"pending"}), "pending session names");
        check(sane((*pending)[0]), "pending value is finite and non-negative");
    }

    timer.reset();
    check(timer.lapCount() == 0, "reset clears the lap list");

    {
        auto lap = timer.lap("fill");
        check(lap.active(), "enabled fill lap is active");
        ctx->fill(d_a, 1u);
    }
    ctx->barrier(d_a, Access::TransferWrite, Access::TransferRead);
    {
        auto lap = timer.lap("copy");
        check(lap.active(), "enabled copy lap is active");
        ctx->copy(d_a, d_b);
    }
    check(timer.lapCount() == 2, "lapCount advances as laps close");

    check(ctx->fsync() == vk::Result::eSuccess, "fsync after the first session succeeds");

    auto laps = timer.read();
    check(laps.ok(), "reading the first session succeeds");
    if (laps.ok()) {
        print_laps(*laps);
        check(names_match(*laps, {"fill", "copy"}), "first session names and order");
        for (const auto& lap : *laps) {
            check(sane(lap), "first session values are finite and non-negative");
        }
    }

    auto second_read = timer.read();
    check(second_read.ok(), "reading twice succeeds");
    if (laps.ok() && second_read.ok()) {
        check(laps->size() == second_read->size(), "repeated read has the same size");
        for (size_t i = 0; i < laps->size() && i < second_read->size(); ++i) {
            check((*laps)[i].ms == (*second_read)[i].ms, "repeated read is idempotent");
        }
    }

    timer.reset();
    check(timer.lapCount() == 0, "reset clears the lap list");

    {
        auto total = timer.lap("total");
        check(total.active(), "nested total lap is active");
        {
            auto child = timer.lap("child_a");
            check(child.active(), "nested child_a lap is active");
            ctx->fill(d_a, 2u);
        }
        ctx->barrier(d_a, Access::TransferWrite, Access::TransferRead);
        {
            auto child = timer.lap("child_b");
            check(child.active(), "nested child_b lap is active");
            ctx->copy(d_a, d_b);
        }
    }

    check(ctx->fsync() == vk::Result::eSuccess, "fsync after the nested session succeeds");

    auto nested = timer.read();
    check(nested.ok(), "reading the nested session succeeds");
    if (nested.ok()) {
        print_laps(*nested);
        check(names_match(*nested, {"total", "child_a", "child_b"}),
              "nested names and creation order");
        check((*nested)[0].ms + 1e-6 >= (*nested)[1].ms, "total covers child_a");
        check((*nested)[0].ms + 1e-6 >= (*nested)[2].ms, "total covers child_b");
        for (const auto& lap : *nested) {
            check(sane(lap), "nested values are finite and non-negative");
        }
    }

    timer.reset();
    {
        auto lap = timer.lap("again");
        check(lap.active(), "reuse lap is active");
        ctx->fill(d_a, 3u);
    }

    auto reuse_pending = timer.read();
    check(not_ready(reuse_pending), "read before the reuse submission completes is eNotReady");
    check(ctx->fsync() == vk::Result::eSuccess, "fsync after the reuse session succeeds");

    auto again = timer.read();
    check(again.ok(), "reading the reuse session succeeds");
    if (again.ok()) {
        print_laps(*again);
        check(names_match(*again, {"again"}), "reuse session names");
        check(sane((*again)[0]), "reuse value is finite and non-negative");
    }
    check(timer.lapCount() == 1, "reuse session lap count");

    timer.reset();
    {
        auto lap = timer.lap("idle");
        check(lap.active(), "wait_idle lap is active");
        ctx->fill(d_b, 4u);
    }
    check(ctx->submit() == vk::Result::eSuccess, "submit before wait_idle succeeds");
    ctx->wait_idle();

    auto idle = timer.read();
    check(idle.ok(), "read after wait_idle succeeds");
    if (idle.ok()) {
        print_laps(*idle);
        check(names_match(*idle, {"idle"}), "wait_idle session names");
        check(sane((*idle)[0]), "wait_idle value is finite and non-negative");
    }
    timer.reset();

    {
        auto first_res  = GpuTimer::create(*ctx, 1);
        auto second_res = GpuTimer::create(*ctx, 1);
        check(first_res.ok() && second_res.ok(), "two timers can share a stream");
        if (first_res.ok() && second_res.ok()) {
            GpuTimer first  = std::move(first_res.val());
            GpuTimer second = std::move(second_res.val());
            {
                auto lap = first.lap("timer_a");
                ctx->fill(d_a, 5u);
            }
            {
                auto lap = second.lap("timer_b");
                ctx->fill(d_b, 6u);
            }
            check(ctx->fsync() == vk::Result::eSuccess, "fsync for both timers succeeds");

            auto first_laps  = first.read();
            auto second_laps = second.read();
            check(first_laps.ok() && second_laps.ok(), "both timers read successfully");
            if (first_laps.ok() && second_laps.ok()) {
                check(names_match(*first_laps, {"timer_a"}), "first timer names");
                check(names_match(*second_laps, {"timer_b"}), "second timer names");
                check(sane((*first_laps)[0]) && sane((*second_laps)[0]),
                      "both timer values are finite and non-negative");
            }
        }
    }

    {
        auto move_res = GpuTimer::create(*ctx, 1);
        check(move_res.ok(), "move test timer is created");
        if (move_res.ok()) {
            GpuTimer source = std::move(move_res.val());
            GpuTimer dest;
            {
                auto lap = source.lap("moved");
                check(lap.active(), "lap survives moving the timer");
                dest = std::move(source);
                check(dest.enabled(), "move destination owns the timer");
                check(lap.active(), "lap remains active after the move");
                ctx->fill(d_a, 7u);
            }
            check(ctx->fsync() == vk::Result::eSuccess, "fsync after the moved lap succeeds");

            auto moved = dest.read();
            check(moved.ok(), "reading the moved timer succeeds");
            if (moved.ok()) {
                print_laps(*moved);
                check(names_match(*moved, {"moved"}), "moved session names");
                check(sane((*moved)[0]), "moved value is finite and non-negative");
            }
        }
    }

    {
        auto acc_res = GpuTimer::create(*ctx, 6);
        check(acc_res.ok(), "accumulating timer is created");
        if (acc_res.ok()) {
            GpuTimer acc = std::move(acc_res.val());
            for (uint32_t b = 0; b < 3; ++b) {
                {
                    auto lap = acc.lap("batch");
                    ctx->fill(d_a, b);
                }
                {
                    auto lap = acc.lap("next");
                    ctx->fill(d_b, b);
                }
                check(ctx->fsync() == vk::Result::eSuccess, "fsync per accumulated batch");
            }

            auto accumulated = acc.read();
            check(accumulated.ok(), "accumulated session reads successfully");
            if (accumulated.ok()) {
                print_laps(*accumulated);
                check(
                    names_match(*accumulated, {"batch", "next", "batch", "next", "batch", "next"}),
                    "accumulated names and creation order");
                for (const auto& lap : *accumulated) {
                    check(sane(lap), "accumulated values are finite and non-negative");
                }
            }
        }
    }

    {
        auto pending_res = GpuTimer::create(*ctx, 1);
        check(pending_res.ok(), "destroy-with-pending timer is created");
        if (pending_res.ok()) {
            GpuTimer pending = std::move(pending_res.val());
            {
                auto lap = pending.lap("destroy_pending");
                ctx->fill(d_a, 8u);
            }
            check(ctx->submit() == vk::Result::eSuccess,
                  "submit before timer destruction succeeds");
        }
    }
    check(ctx->fsync() == vk::Result::eSuccess, "fsync after destroy-with-pending succeeds");

    {
        Stream* stream = ctx->get_stream(cDefaultStream);
        check(stream && stream->begin() == vk::Result::eSuccess,
              "the stream begins a pre-existing recording");
        ctx->fill(d_a, 13u);
        check(stream && stream->end() == vk::Result::eSuccess,
              "the pre-existing recording ends without submit");

        auto fresh_res = GpuTimer::create(*ctx, 1);
        check(fresh_res.ok(), "timer created after the pre-existing discard");
        if (fresh_res.ok()) {
            GpuTimer fresh = std::move(fresh_res.val());
            {
                auto lap = fresh.lap("after_discard");
                check(lap.active(), "the first lap absorbs the pre-existing discard");
                ctx->fill(d_a, 14u);
            }
            check(ctx->fsync() == vk::Result::eSuccess,
                  "fsync after the fresh discard session succeeds");

            auto fresh_laps = fresh.read();
            check(fresh_laps.ok(), "read succeeds after absorbing a pre-existing discard");
            if (fresh_laps.ok()) {
                check(names_match(*fresh_laps, {"after_discard"}), "fresh discard session names");
                check(sane((*fresh_laps)[0]), "fresh discard value is finite and non-negative");
            }
        }
    }

    {
        auto discard_res = GpuTimer::create(*ctx, 2);
        check(discard_res.ok(), "discard timer is created");
        if (discard_res.ok()) {
            GpuTimer discarded_timer = std::move(discard_res.val());
            {
                auto lap = discarded_timer.lap("discarded");
                check(lap.active(), "discarded lap is active");
                ctx->fill(d_a, 10u);
            }
            Stream* stream = ctx->get_stream(cDefaultStream);
            check(stream && stream->end() == vk::Result::eSuccess,
                  "ending the command buffer without submit succeeds");

            ctx->fill(d_a, 11u);
            check(ctx->fsync() == vk::Result::eSuccess,
                  "a later submission completes after the discard");

            auto discarded = discarded_timer.read();
            check(!discarded.ok() && discarded.err() == vk::Result::eErrorUnknown,
                  "a discarded recording invalidates the session");

            discarded_timer.reset();
            {
                auto lap = discarded_timer.lap("recovered");
                check(lap.active(), "a lap after discard recovery is active");
                ctx->fill(d_a, 12u);
            }
            check(ctx->fsync() == vk::Result::eSuccess, "fsync after discard recovery succeeds");

            auto recovered = discarded_timer.read();
            check(recovered.ok(), "reading the recovered session succeeds");
            if (recovered.ok()) {
                check(names_match(*recovered, {"recovered"}), "recovered session names");
                check(sane((*recovered)[0]), "recovered value is finite and non-negative");
            }
        }
    }

#ifdef NDEBUG
    {
        auto poison_res = GpuTimer::create(*ctx, 1);
        check(poison_res.ok(), "epoch poison timer is created");
        if (poison_res.ok()) {
            GpuTimer poisoned = std::move(poison_res.val());
            {
                auto lap = poisoned.lap("spanning");
                check(lap.active(), "spanning lap is active");
                ctx->fill(d_a, 9u);
                check(ctx->submit() == vk::Result::eSuccess,
                      "submit in the middle of a lap succeeds");
            }
            auto bad = poisoned.read();
            check(!bad.ok() && bad.err() == vk::Result::eErrorUnknown,
                  "a lap spanning a submit poisons the session");

            poisoned.reset();
            auto still_bad = poisoned.read();
            check(!still_bad.ok() && still_bad.err() == vk::Result::eErrorUnknown,
                  "reset cannot clear an in-flight failed-close session");

            check(ctx->fsync() == vk::Result::eSuccess,
                  "fsync after the poisoned session succeeds");

            poisoned.reset();
            auto recovered = poisoned.read();
            check(recovered.ok() && recovered->empty(),
                  "reset recovers after the failed-close submission completed");
        }
    }

    {
        const StreamHandle cap_stream = ctx->create_stream();
        auto cap_res                  = GpuTimer::create(*ctx, 1, cap_stream);
        check(cap_res.ok(), "capacity timer is created");
        if (cap_res.ok()) {
            GpuTimer& capacity = *cap_res;
            {
                auto lap = capacity.lap("one");
                check(lap.active(), "first capacity lap is active");
                auto overflow = capacity.lap("overflow");
                check(!overflow.active(), "capacity overflow poisons the session");
            }
            auto bad = capacity.read();
            check(!bad.ok() && bad.err() == vk::Result::eErrorUnknown,
                  "reading a poisoned session returns an error");
        }
        check(ctx->submit(cap_stream) == vk::Result::eSuccess,
              "the capacity session submits cleanly");
    }
#endif

    ctx->deallocate(d_a);
    ctx->deallocate(d_b);

    std::printf("%s\n", g_ok ? "gpu timer test passed" : "GPU TIMER TEST FAILED");
    return g_ok ? 0 : 1;
}
