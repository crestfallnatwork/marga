/* SPDX-License-Identifier: MPL-2.0 */
/* Author: Himanshu Gupta <himanshu3@iisc.ac.in> */

#pragma once

#include "pictor_core/execution/synchronization.hpp"
#include "pictor_core/memory/allocation_type.hpp"

#include <vulkan/vulkan.hpp>

#include <array>
#include <vector>

namespace pictor {

class Context;

using StreamHandle                           = uint32_t;
inline constexpr StreamHandle cDefaultStream = 0;

using RecordCallback = void (*)(vk::CommandBuffer cmd, void* user_data);

class Stream {
  public:
    Stream(vk::Device device, uint32_t queue_family_index, float timestamp_period = 1.0f);
    ~Stream();

    Stream(const Stream&)            = delete;
    Stream& operator=(const Stream&) = delete;

    vk::Result begin();
    vk::Result end();
    vk::Result flush(vk::Queue queue);

    void synchronize();
    vk::Result fsync(vk::Queue queue);

    void barrier(const AllocationInfo& resource, Access from, Access to);
    void barrier(const std::vector<ResourceBarrier>& barriers);

    void record(RecordCallback cb, void* user_data);

    // GPU timestamp instrumentation (optional; used for ablation attribution).
    // write_timestamp records a device timestamp into `slot` at the current
    // command-buffer position; read_timestamps blocks until the results are
    // available (call only after synchronize/fsync) and returns nanoseconds.
    void reset_timestamps(uint32_t first, uint32_t count);
    void write_timestamp(uint32_t slot);
    std::vector<uint64_t> read_timestamps(uint32_t first, uint32_t count);

    vk::Device device;
    vk::CommandPool command_pool;

    vk::CommandBuffer current_command_buffer() const
    {
        return _command_buffers[_current_frame];
    }

  private:
    friend class Context;

    void _copy(const AllocationInfo& src, const AllocationInfo& dst);
    void _fill(const AllocationInfo& dst, size_t offset, size_t size, uint32_t value);
    void _update(const AllocationInfo& dst, size_t offset, size_t size, const void* data);

    std::array<vk::CommandBuffer, 2> _command_buffers;
    std::array<vk::Fence, 2> _fences;
    vk::QueryPool _timestamp_pool;
    float _timestamp_period{1.0f}; // ns per timestamp tick
    uint32_t _current_frame{0};
    bool _recording{false};
};

} // namespace pictor
