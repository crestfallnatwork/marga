# In-app GPU phase timing with `GpuTimer`

`pictor::GpuTimer` measures how long GPU work takes between two timestamp queries recorded on one
stream. It answers "which phase of this submission is slow" without launching an external profiler;
use Nsight Systems or GPU Trace when you need the full timeline, API trace or hardware counters
(see [profiling.md](profiling.md)).

A lap is an RAII scope around recorded work. Construction records a start timestamp, destruction
records an end timestamp, so a timed phase is one line and cannot be left half-closed:

```cpp
#include "pictor_core/execution/gpu_timer.hpp"

GpuTimer timer; // disabled until create() succeeds
if (auto created = GpuTimer::create(*ctx, 3); created.ok()) {
    timer = std::move(*created);
} else {
    std::fprintf(stderr, "GpuTimer unavailable: %d\n", static_cast<int>(created.err()));
}

{
    auto lap = timer.lap("fill");
    ctx->fill(buffer, 0u);
}
{
    auto lap = timer.lap("copy");
    ctx->copy(src, dst);
}

ctx->fsync();

auto laps = timer.read();
if (laps.ok()) {
    for (const auto& lap : *laps) {
        std::printf("%.*s: %.3f ms\n", static_cast<int>(lap.name.size()), lap.name.data(),
                    lap.ms);
    }
}
```

Always bind the guard to a named local (`auto lap = timer.lap("phase");`). A temporary such as
`timer.lap("phase");` closes at the end of the statement and measures almost nothing.

## Disabled timers

`GpuTimer::create(ctx, 0)` and a default-constructed `GpuTimer` are disabled no-ops: `lap()`
returns an inactive guard, nothing is recorded and nothing is allocated. That lets a profiling flag
stay out of the phase code:

```cpp
const bool profiling = std::getenv("MARGA_TC_TIMESTAMPS") != nullptr;

GpuTimer timer; // disabled when profiling is off
if (auto created = GpuTimer::create(*ctx, profiling ? 4u * num_batches : 0u); created.ok()) {
    timer = std::move(*created);
} else {
    std::fprintf(stderr, "GpuTimer unavailable: %d\n", static_cast<int>(created.err()));
}
```

A disabled `read()` is an **empty success**. Fixed-stride consumers must therefore check the span
size before indexing:

```cpp
if (laps.ok() && laps->size() == 4u * num_batches) {
    // safe to index in strides of four
}
```

## What it measures

A lap measures GPU execution between its two timestamp writes. It does **not** measure host time:
recording, `submit`, `fsync`, queue wait and host readback all happen outside the timestamps. If you
also need the host cost, keep a host clock around the record/wait calls:

```cpp
const auto record_start = std::chrono::high_resolution_clock::now();
// record work and close laps
const auto record_end = std::chrono::high_resolution_clock::now();

ctx->fsync();
const auto wait_end = std::chrono::high_resolution_clock::now();
```

Vulkan events are synchronization primitives and cannot report elapsed time; timestamp queries are
the mechanism behind `GpuTimer`.

## Sessions and capacity

A **session** starts at `GpuTimer::create()` or `reset()` and ends at the next `reset()` (or timer
destruction). A session may span several submissions: close each lap before `submit()`/`fsync()`,
then record more laps in the next command buffer. `max_laps` bounds the whole session, not one
submission. For example, the marga loop below uses `4 * num_batches` laps across `num_batches`
submissions and reads once at the end.

`reset()` starts a new session after the previous one has been submitted and completed
(`fsync()`); it clears the host lap list and arms a GPU-side query reset for the next `lap()`. Read
the previous results before resetting if you want to keep them. Resetting with open laps, or with
unsubmitted/in-flight laps from the previous session, asserts in debug and poisons the session in
release.

Each lap costs two timestamp queries. Exceeding `max_laps` is a programming error: debug asserts,
release poisons the session and drops further laps, and `read()` returns an error.

## Marga four-phase batch

The batch loop records one parent lap and the three phases it contains. All laps close before the
per-batch `fsync()`:

```cpp
uint64_t total_count = 0;

GpuTimer timer; // disabled unless create succeeds
if (auto created = GpuTimer::create(*ctx, 4u * num_batches); created.ok()) {
    timer = std::move(*created);
} else {
    std::fprintf(stderr, "GpuTimer unavailable: %d\n", static_cast<int>(created.err()));
}

for (uint64_t b = 0, edge_start = 0; edge_start < g.num_edges;
     edge_start += cBatchSize, ++b) {
    {
        auto batch = timer.lap("batch");
        {
            auto lap = timer.lap("draw");
            tc->record_raw(tx, ty, 1, params);
        }
        {
            auto lap = timer.lap("copy");
            ctx->barrier(result, Access::Write, Access::TransferRead);
            ctx->copy(result, h_result);
        }
        {
            auto lap = timer.lap("fill");
            ctx->barrier(result, Access::TransferRead, Access::TransferWrite);
            ctx->fill(result, 0);
        }
    }

    ctx->fsync();
    ctx->invalidate(h_result);
    total_count += *res;
}

auto laps = timer.read();
if (laps.ok() && laps->size() == 4u * num_batches) {
    for (uint64_t b = 0; b < num_batches; ++b) {
        const auto* row = laps->data() + 4 * b;
        std::printf("TSBATCH batch=%llu draw_us=%.1f copy_us=%.1f fill_us=%.1f total_us=%.1f\n",
                    static_cast<unsigned long long>(b), row[1].ms * 1e3, row[2].ms * 1e3,
                    row[3].ms * 1e3, row[0].ms * 1e3);
    }
}
```

The parent `batch` and its children overlap, so do not add all four values together. Use the parent
as the total and the children as the breakdown. The three child laps alone are the faithful
replacement for the old four shared marks; the parent is an optional total and costs two extra
timestamp writes per batch.

## BFS multi-level

A BFS batch records several levels into one submission. The parent `level` lap covers any gaps
between the child phases:

```cpp
constexpr uint32_t kLevelsPerBatch = 10;

GpuTimer timer; // disabled unless create succeeds
if (auto created = GpuTimer::create(*ctx, 5u * kLevelsPerBatch); created.ok()) {
    timer = std::move(*created);
} else {
    std::fprintf(stderr, "GpuTimer unavailable: %d\n", static_cast<int>(created.err()));
}

for (uint32_t batch_start = 0; batch_start < levels; batch_start += kLevelsPerBatch) {
    const uint32_t batch_end = std::min(levels, batch_start + kLevelsPerBatch);

    for (uint32_t level = batch_start; level < batch_end; ++level) {
        auto level_lap = timer.lap("level");

        {
            auto lap = timer.lap("task");
            grid->record(grid_params);
        }
        {
            auto lap = timer.lap("fill");
            ctx->fill(frontier_sizes[next], 0);
            ctx->barrier(frontier_sizes[next], Access::TransferWrite, Access::Read);
        }
        {
            auto lap = timer.lap("mesh");
            bfs->record_indirect(indirect, bfs_params);
        }
        {
            auto lap = timer.lap("post");
            ctx->barrier(frontier_sizes[next], Access::Write, Access::Read);
        }
    }

    ctx->fsync();

    auto laps = timer.read();
    if (!laps.ok()) {
        std::fprintf(stderr, "GpuTimer read failed: %d\n", static_cast<int>(laps.err()));
        return false;
    }
    for (const auto& lap : *laps) {
        accumulate_timing(lap.name, lap.ms);
    }
    timer.reset();
}
```

As in the batch example, `level` overlaps its children; sum only the child laps.

## Cost

Every lap writes two timestamps, both with `vk::PipelineStageFlagBits2::eAllCommands`. That stage is
the right general default for compute, mesh and transfer phases, but it is an ordering point: each
mark waits for all previously recorded work. A timestamp write is not free even when it does not
stall; budget on the order of hundreds of nanoseconds per boundary. The pair-based design doubles
the timestamp writes relative to a shared-boundary mark API:

- marga batch: 6 writes for the three phases, 8 with the optional parent;
- BFS level: 10 writes for the five laps (parent + four phases).

For the observed 0.16-0.34 ms marga phases the overhead is negligible. For very short phases or
tightly pipelined loops, measure the inflation or use fewer laps. The old benchmark harness used
`eComputeShader` while `GpuTimer` uses `eAllCommands`, so benchmark numbers from before and after
this change are not directly comparable; re-baseline.

## Contracts

- **One stream per timer.** The stream is chosen at creation; `lap()` has no stream argument.
  Multiple timers may share one stream. Timestamps from different queues are not comparable without
  calibrated timestamps.
- **A lap may not span a submit.** Close every lap before `submit()`/`fsync()`. A lap that crosses a
  command-buffer boundary is detected through the stream's recording epoch: debug builds assert,
  and release builds poison the session so `read()` reports an error instead of publishing a stray
  timestamp.
- **Read after submission completed.** `read()` is non-blocking. It returns `vk::Result::eNotReady`
  without touching Vulkan while any lap is unsubmitted or still in flight, so it is VUID-safe to
  call after closing laps but before `fsync()` **when the laps' command buffer was submitted**. A
  discarded recording (a command buffer ended without submit, or a failed submit) invalidates the
  session: `read()` returns `vk::Result::eErrorUnknown` instead of querying those queries. Call
  `read()` after `ctx->fsync(stream)` or `ctx->wait_idle()` for results. It also requires that every
  lap is closed; an open lap or a poisoned session returns `vk::Result::eErrorUnknown` (and asserts
  in debug).
- **Discarded recordings.** A command buffer ended without submitting is discarded when the stream
  next begins recording. If that happens before the timer's first lap (for example, a pre-existing
  abandoned recording), the first `lap()` starts a fresh command buffer and absorbs the discard. If
  it happens after the session has recorded a lap, the session is invalid: further `lap()` calls are
  inactive and `read()` returns `vk::Result::eErrorUnknown`. Recreate the timer, or submit a later
  command buffer and `reset()` after it completes before starting a new session.
- **Reset after completion.** `reset()` requires the previous session to be submitted and
  completed. It is `noexcept`, performs no Vulkan call, and does not invalidate a span already
  returned by `read()`.
- **Capacity is per session.** Size `max_laps` to the whole session (for example
  `4 * num_batches` when accumulating, or `5 * levels_per_submission` when resetting per batch).
- **Names are views.** `LapResult::name` points at the `std::string_view` passed to `lap()`. The
  characters must outlive the last use of any span returned by `read()`; string literals are the
  intended use.
- **Results are flat.** Laps appear in creation order. Nested and overlapping laps are supported,
  but the timer does not build a tree.
- **Spans are timer-owned.** A span from `read()` is invalidated by the next `read()` and by timer
  destruction, not by `reset()`. Copy the values if you need them later.
- **Move rules.** `GpuTimer` is move-only; a moved-from timer is disabled. A live `Lap` survives a
  timer move because it references the implementation. Do not move-assign over a timer that still
  has open laps (debug assert).
- **Lifetime.** Destroy every `Lap` before its timer, and every timer before its `Context`.
  `~GpuTimer` waits for the timer's submitted work before destroying the query pool (and skips the
  wait when no timestamp was recorded), but it cannot repair a discarded, failed or unsubmitted
  recording: submit (or discard the command buffer) before destroying a timer that recorded laps.
  Do not destroy a timer while a recording that references its pool is still pending.
- **No internal locking.** A timer follows the stream's single-recording-thread model.
- **Render passes.** The first `lap()` after `create()`/`reset()` records `vkCmdResetQueryPool`,
  which is illegal inside a render pass; start that lap outside a pass. Multiview timestamps write
  several queries at once and are not supported by the two-queries-per-lap layout.
- **Timestamps are optional.** `create` returns `vk::Result::eErrorFeatureNotPresent` when the
  selected queue family cannot write timestamps (`timestampValidBits == 0` or a non-positive/NaN
  `timestampPeriod`); `max_laps == 0` is exempt and always succeeds as a disabled timer. It returns
  `vk::Result::eErrorInvalidExternalHandle` for an unknown stream and
  `vk::Result::eErrorInitializationFailed` when `max_laps` exceeds the query index range.

## Relation to the external tools

Use `GpuTimer` for in-engine phase attribution that ships with the workload (or is gated by an
environment variable). Use Nsight Systems for the host/GPU timeline, submit boundaries and API
trace, and Nsight Graphics GPU Trace for hardware counters and shader analysis. The two views are
complementary: `GpuTimer` tells you which phase got slower; the tools tell you why.
