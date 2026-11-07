/* SPDX-License-Identifier: MPL-2.0 */

#include "pictor_core/execution/gpu_timer.hpp"

#include "pictor_core/context/context.hpp"

#include <vulkan/vulkan.hpp>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

namespace pictor {

struct GpuTimer::Impl {
    struct LapRecord {
        std::string_view name;
        uint64_t epoch{0};
    };

    vk::Device device;
    Stream* stream{nullptr};
    vk::QueryPool pool;

    uint32_t max_laps{0};
    uint64_t valid_mask{0};
    double ms_per_tick{0.0};

    std::vector<LapRecord> laps;
    std::vector<uint64_t> ticks;
    std::vector<LapResult> results;

    uint32_t lap_count{0};
    uint32_t open_laps{0};
    uint64_t last_epoch{0};
    uint64_t discard_seq{0};
    vk::Result failure{vk::Result::eSuccess};
    bool needs_reset{false};
    bool poisoned{false};

    ~Impl()
    {
        if (stream) {
            assert(open_laps == 0 && "GpuTimer destroyed with an open Lap");
            assert((lap_count == 0 || last_epoch <= stream->_submitted_epoch) &&
                   "GpuTimer destroyed with an unsubmitted recording");
            if (last_epoch != 0 && last_epoch <= stream->_submitted_epoch &&
                last_epoch > stream->_completed_epoch) {
                stream->_synchronize_submitted();
            }
        }
        if (pool) {
            device.destroyQueryPool(pool);
        }
    }
};

namespace {

struct TimestampCmd {
    vk::QueryPool pool;
    uint32_t query;
};

struct QueryResetCmd {
    vk::QueryPool pool;
    uint32_t count;
};

void write_timestamp(vk::CommandBuffer cmd, void* data)
{
    const auto* c = static_cast<const TimestampCmd*>(data);
    cmd.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, c->pool, c->query);
}

void reset_query_pool(vk::CommandBuffer cmd, void* data)
{
    const auto* c = static_cast<const QueryResetCmd*>(data);
    cmd.resetQueryPool(c->pool, 0, c->count);
}

} // namespace

GpuTimer::GpuTimer() noexcept = default;

GpuTimer::GpuTimer(std::unique_ptr<Impl> impl) noexcept : _impl(std::move(impl)) {}

GpuTimer::~GpuTimer() = default;

GpuTimer::GpuTimer(GpuTimer&&) noexcept = default;

GpuTimer& GpuTimer::operator=(GpuTimer&& other) noexcept
{
    if (this != &other) {
        assert((!_impl || _impl->open_laps == 0) && "GpuTimer move-assigned over an open Lap");
        _impl = std::move(other._impl);
    }
    return *this;
}

Expected<GpuTimer, vk::Result> GpuTimer::create(Context& ctx, uint32_t max_laps,
                                                StreamHandle stream)
{
    if (max_laps == 0) {
        return GpuTimer{};
    }
    if (max_laps > std::numeric_limits<uint32_t>::max() / 2) {
        assert(false && "GpuTimer capacity exceeds the query index range");
        return vk::Result::eErrorInitializationFailed;
    }

    Stream* target = ctx.get_stream(stream);
    if (!target) {
        return vk::Result::eErrorInvalidExternalHandle;
    }

    const DeviceInfo& info = ctx.device_info;
    if (info.timestampValidBits == 0 || !(info.timestampPeriod > 0.0f)) {
        return vk::Result::eErrorFeatureNotPresent;
    }

    auto impl      = std::make_unique<Impl>();
    impl->max_laps = max_laps;
    impl->valid_mask =
        info.timestampValidBits >= 64 ? ~0ull : ((1ull << info.timestampValidBits) - 1ull);
    impl->ms_per_tick = static_cast<double>(info.timestampPeriod) / 1e6;
    impl->laps.resize(max_laps);
    impl->ticks.resize(static_cast<size_t>(2) * max_laps);
    impl->results.resize(max_laps);

    vk::QueryPoolCreateInfo pool_info{};
    pool_info.queryType  = vk::QueryType::eTimestamp;
    pool_info.queryCount = 2 * max_laps;

    auto pool_res = ctx.device.createQueryPool(pool_info);
    if (pool_res.result != vk::Result::eSuccess) {
        return pool_res.result;
    }

    impl->device      = ctx.device;
    impl->stream      = target;
    impl->pool        = pool_res.value;
    impl->discard_seq = target->_discard_seq;
    impl->needs_reset = true;

    return GpuTimer(std::move(impl));
}

bool GpuTimer::enabled() const noexcept
{
    return static_cast<bool>(_impl);
}

uint32_t GpuTimer::lapCount() const noexcept
{
    return _impl ? _impl->lap_count : 0u;
}

GpuTimer::Lap GpuTimer::lap(std::string_view name) noexcept
{
    if (!_impl) {
        return Lap{};
    }
    return _open_lap(name);
}

GpuTimer::Lap GpuTimer::_open_lap(std::string_view name) noexcept
{
    Impl& s = *_impl;
    if (s.poisoned || s.failure != vk::Result::eSuccess) {
        return Lap{};
    }
    if (s.lap_count + s.open_laps >= s.max_laps) {
        assert(false && "GpuTimer capacity exceeded");
        s.poisoned = true;
        return Lap{};
    }

    Stream* stream = s.stream;
    if (!stream) {
        s.failure = vk::Result::eErrorUnknown;
        return Lap{};
    }

    const bool pristine = s.lap_count == 0 && s.open_laps == 0 && s.last_epoch == 0;
    if (stream->_discard_seq != s.discard_seq) {
        if (!pristine) {
            return Lap{};
        }
        s.discard_seq = stream->_discard_seq;
    }
    if (!stream->_recording) {
        const vk::Result begin_res = stream->begin();
        if (begin_res != vk::Result::eSuccess) {
            s.failure = begin_res;
            return Lap{};
        }
        if (stream->_discard_seq != s.discard_seq) {
            if (!pristine) {
                return Lap{};
            }
            s.discard_seq = stream->_discard_seq;
        }
    }

    if (s.needs_reset) {
        QueryResetCmd reset{s.pool, 2 * s.max_laps};
        stream->record(reset_query_pool, &reset);
        s.needs_reset = false;
    }

    const uint32_t index = s.lap_count + s.open_laps;
    TimestampCmd start{s.pool, 2 * index};
    stream->record(write_timestamp, &start);

    const uint64_t epoch = stream->_epoch;
    s.laps[index]        = {name, epoch};
    s.last_epoch         = std::max(s.last_epoch, epoch);
    ++s.open_laps;

    return {&s, index};
}

void GpuTimer::_close_lap(Impl* impl, uint32_t index) noexcept
{
    if (!impl || index >= impl->max_laps) {
        return;
    }

    Impl& s = *impl;
    if (s.open_laps == 0) {
        return;
    }
    --s.open_laps;

    Impl::LapRecord& record = s.laps[index];
    if (!s.stream || !s.stream->_recording || s.stream->_epoch != record.epoch) {
        assert(false && "GpuTimer lap spans a submit or command-buffer end");
        s.poisoned = true;
        return;
    }

    TimestampCmd end{s.pool, (2 * index) + 1};
    s.stream->record(write_timestamp, &end);

    ++s.lap_count;
}

void GpuTimer::Lap::_close() noexcept
{
    if (!_active) {
        return;
    }
    _active    = false;
    Impl* impl = _impl;
    _impl      = nullptr;
    if (impl) {
        GpuTimer::_close_lap(impl, _index);
    }
}

GpuTimer::Lap::Lap(Impl* impl, uint32_t index) noexcept : _impl(impl), _index(index), _active(true)
{
}

GpuTimer::Lap::~Lap()
{
    _close();
}

GpuTimer::Lap::Lap(Lap&& other) noexcept
    : _impl(other._impl), _index(other._index), _active(other._active)
{
    other._impl   = nullptr;
    other._index  = 0;
    other._active = false;
}

GpuTimer::Lap& GpuTimer::Lap::operator=(Lap&& other) noexcept
{
    if (this != &other) {
        _close();
        _impl         = other._impl;
        _index        = other._index;
        _active       = other._active;
        other._impl   = nullptr;
        other._index  = 0;
        other._active = false;
    }
    return *this;
}

bool GpuTimer::Lap::active() const noexcept
{
    return _active;
}

void GpuTimer::reset() noexcept
{
    if (!_impl) {
        return;
    }

    Impl& s = *_impl;
    if (s.open_laps != 0) {
        assert(false && "GpuTimer::reset with an open lap");
        s.poisoned = true;
        return;
    }
    if (s.stream && s.last_epoch > s.stream->_completed_epoch) {
        assert(false && "GpuTimer::reset requires the prior session to be submitted and completed");
        s.poisoned = true;
        return;
    }

    s.lap_count   = 0;
    s.open_laps   = 0;
    s.last_epoch  = 0;
    s.discard_seq = s.stream ? s.stream->_discard_seq : 0;
    s.failure     = vk::Result::eSuccess;
    s.needs_reset = true;
    s.poisoned    = false;
}

Expected<std::span<const GpuTimer::LapResult>, vk::Result> GpuTimer::read()
{
    if (!_impl) {
        return std::span<const LapResult>{};
    }

    Impl& s = *_impl;
    if (s.failure != vk::Result::eSuccess) {
        return s.failure;
    }
    if (s.open_laps != 0 || s.poisoned) {
        assert(false && "GpuTimer::read with an open lap or a poisoned session");
        return vk::Result::eErrorUnknown;
    }
    if (s.lap_count == 0) {
        return std::span<const LapResult>{};
    }
    if (!s.stream) {
        return vk::Result::eErrorUnknown;
    }
    if (s.stream->_discard_seq != s.discard_seq) {
        return vk::Result::eErrorUnknown;
    }
    for (uint32_t i = 0; i < s.lap_count; ++i) {
        if (s.laps[i].epoch > s.stream->_completed_epoch) {
            return vk::Result::eNotReady;
        }
    }

    const uint32_t query_count = 2 * s.lap_count;
    const vk::Result res       = s.device.getQueryPoolResults(
        s.pool, 0, query_count, static_cast<vk::DeviceSize>(query_count) * sizeof(uint64_t),
        s.ticks.data(), sizeof(uint64_t), vk::QueryResultFlagBits::e64);
    if (res != vk::Result::eSuccess) {
        return res;
    }

    for (uint32_t i = 0; i < s.lap_count; ++i) {
        const uint64_t start = s.ticks[2 * i];
        const uint64_t end   = s.ticks[(2 * i) + 1];
        const uint64_t delta = (end - start) & s.valid_mask;
        s.results[i]         = {s.laps[i].name, static_cast<double>(delta) * s.ms_per_tick};
    }

    return std::span<const LapResult>(s.results.data(), s.lap_count);
}

} // namespace pictor
