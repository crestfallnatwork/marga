/* SPDX-License-Identifier: MPL-2.0 */

#pragma once

#include "pictor_core/context/result_type.hpp"
#include "pictor_core/execution/stream.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <string_view>

namespace pictor {

class Context;

// GPU-side phase timing on one Stream. Every Lap owns an independent start/end timestamp pair, so
// laps may be sequential, nested or overlapping; results are flat, in creation order. A disabled
// timer (default-constructed or create(ctx, 0)) records nothing and allocates nothing.
//
// Move-only. A moved-from timer is disabled. Moving a timer with live Lap guards is safe because a
// guard references the implementation, not the timer; move-assigning over a timer that still has
// open laps is undefined in release (debug assert). ~GpuTimer waits for the timer's submitted work
// before destroying the pool; it cannot repair a discarded, failed or unsubmitted recording, so
// submit or discard such a recording before destroying the timer.
class GpuTimer {
    struct Impl;

  public:
    struct LapResult {
        std::string_view name;
        double ms{0.0};
    };

    GpuTimer() noexcept;
    ~GpuTimer();

    GpuTimer(GpuTimer&& other) noexcept;
    GpuTimer& operator=(GpuTimer&& other) noexcept;
    GpuTimer(const GpuTimer&)            = delete;
    GpuTimer& operator=(const GpuTimer&) = delete;

    // max_laps bounds one session (create()/reset() until the next reset()). Returns
    // eErrorFeatureNotPresent when timestamps are unsupported, eErrorInvalidExternalHandle for an
    // unknown stream, and eErrorInitializationFailed when max_laps exceeds the query index range.
    // max_laps == 0 returns a disabled timer without touching the device.
    static Expected<GpuTimer, vk::Result> create(Context& ctx, uint32_t max_laps,
                                                 StreamHandle stream = cDefaultStream);

    [[nodiscard]] bool enabled() const noexcept;

    // Closed laps in the current session. Zero when disabled or before the first lap closes.
    [[nodiscard]] uint32_t lapCount() const noexcept;

    // RAII lap: start timestamp on construction, end timestamp on destruction. Bind the guard to a
    // named local (`auto lap = timer.lap("phase");`); a temporary closes at the end of the
    // statement and measures almost nothing. Nests naturally; disabled timer -> inactive guard.
    class Lap {
      public:
        Lap() noexcept = default;
        ~Lap();
        Lap(Lap&& other) noexcept;
        // Move-assigning over an active guard closes the destination's lap first.
        Lap& operator=(Lap&& other) noexcept;
        Lap(const Lap&)            = delete;
        Lap& operator=(const Lap&) = delete;
        [[nodiscard]] bool active() const noexcept;

      private:
        friend class GpuTimer;
        Lap(Impl* impl, uint32_t index) noexcept;
        void _close() noexcept;
        Impl* _impl{nullptr};
        uint32_t _index{0};
        bool _active{false};
    };

    [[nodiscard]] Lap lap(std::string_view name) noexcept;

    // Starts a new session and arms a GPU-side query reset for the next lap(). Requires the
    // previous session to be submitted and completed (ctx->fsync()); it performs no Vulkan call.
    void reset() noexcept;

    // Non-blocking. Requires every lap's submission to have completed: call after
    // ctx->fsync(stream) / wait_idle(). Returns eNotReady without touching Vulkan while the laps
    // are unsubmitted or in flight. Returns eErrorUnknown for an open lap, a poisoned session, or a
    // session whose recording was discarded; a runtime begin failure is returned as its vk::Result.
    // The span is invalidated by the next read() (not by reset()) and by timer destruction.
    [[nodiscard]] Expected<std::span<const LapResult>, vk::Result> read();

  private:
    explicit GpuTimer(std::unique_ptr<Impl> impl) noexcept;
    Lap _open_lap(std::string_view name) noexcept;
    static void _close_lap(Impl* impl, uint32_t index) noexcept;
    std::unique_ptr<Impl> _impl;
};

} // namespace pictor
