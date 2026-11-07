/* SPDX-License-Identifier: MPL-2.0 */

#pragma once

#include "harness.hpp"

#include "pictor_core/context/context.hpp"
#include "pictor_core/execution/gpu_timer.hpp"

#include <cassert>
#include <chrono>
#include <vector>

namespace pictor::bench {

using Clock = std::chrono::steady_clock;

// Transfer chunk length, in elements.
inline constexpr uint64_t cChunk = 1u << 21u;

// Copies `bytes` bytes from `src` at byte offset `src_offset` to `dst` at byte offset `dst_offset`,
// then waits.
inline void copy_range(Context& ctx, const AllocationInfo& src, uint64_t src_offset,
                       const AllocationInfo& dst, uint64_t dst_offset, uint64_t bytes)
{
    AllocationInfo s = src;
    s.offset         = src_offset;
    s.size           = bytes;
    AllocationInfo d = dst;
    d.offset         = dst_offset;
    d.size           = bytes;
    ctx.copy(s, d);
    ctx.fsync();
}

// Reads `bytes` bytes from `src` at byte offset `src_offset` into the start of host buffer `dst`,
// then invalidates `dst` for reading.
inline void copy_range_to_host(Context& ctx, const AllocationInfo& src, uint64_t src_offset,
                               AllocationInfo& dst, uint64_t bytes)
{
    copy_range(ctx, src, src_offset, dst, 0, bytes);
    ctx.invalidate(dst);
}

// Runs `warmup` warmup records, then times `iters` records with device timestamps. Returns the
// median in ms. The host-clock fallback is reached when the device cannot create a timer, or
// (rarely) when the timer readback failed after the timed loop; in the latter case the timed loop
// is deliberately re-recorded so the returned sample is still host-measured.
template <typename RecordFn>
double measure_stream(Context& ctx, int warmup, int iters, RecordFn&& record_workload)
{
    if (iters <= 0)
        return 0.0;

    for (int w = 0; w < warmup; w++)
        record_workload();
    ctx.fsync();

    std::vector<double> per_iter;

    auto timer_res = GpuTimer::create(ctx, static_cast<uint32_t>(iters));
    if (timer_res.ok()) {
        GpuTimer& timer = *timer_res;
        for (int i = 0; i < iters; i++) {
            auto lap = timer.lap("iter");
            record_workload();
        }
        ctx.fsync();

        auto laps = timer.read();
        if (laps.ok()) {
            assert(laps->size() == static_cast<size_t>(iters));
            if (laps->size() == static_cast<size_t>(iters)) {
                per_iter.reserve(laps->size());
                for (const auto& lap : *laps) {
                    per_iter.push_back(lap.ms);
                }
            }
        }
    }

    if (per_iter.empty()) {
        auto t0 = Clock::now();
        for (int i = 0; i < iters; i++)
            record_workload();
        ctx.fsync();
        auto t1 = Clock::now();
        per_iter.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count() / iters);
    }

    return median(per_iter);
}

} // namespace pictor::bench
