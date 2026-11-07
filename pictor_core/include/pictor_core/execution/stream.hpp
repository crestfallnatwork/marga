/* SPDX-License-Identifier: MPL-2.0 */
/* Author: Himanshu Gupta <himanshu3@iisc.ac.in> */

#pragma once

#include "pictor_core/execution/synchronization.hpp"
#include "pictor_core/memory/allocation_type.hpp"

#include <vulkan/vulkan.hpp>

#include <array>
#include <cstdint>
#include <string>

namespace pictor {

class Context;
class GpuTimer;

using StreamHandle                           = uint32_t;
inline constexpr StreamHandle cDefaultStream = 0;

using RecordCallback = void (*)(vk::CommandBuffer cmd, void* user_data);

class Stream {
  public:
    Stream(vk::Device device, uint32_t queue_family_index);
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

    void begin_label(const char* name);
    void end_label();
    void insert_label(const char* name);

    void record(RecordCallback cb, void* user_data);
    void record(const char* label, RecordCallback cb, void* user_data);

    vk::Device device;
    vk::CommandPool command_pool;

    [[nodiscard]] vk::CommandBuffer current_command_buffer() const
    {
        return _command_buffers[_current_frame];
    }

  private:
    friend class Context;
    friend class GpuTimer;

    void _set_debug_name(StreamHandle handle);
    void _synchronize_submitted();

    void _copy(const AllocationInfo& src, const AllocationInfo& dst);
    void _fill(const AllocationInfo& dst, size_t offset, size_t size, uint32_t value);
    void _update(const AllocationInfo& dst, size_t offset, size_t size, const void* data);

    std::array<vk::CommandBuffer, 2> _command_buffers;
    std::array<vk::Fence, 2> _fences;
    std::array<bool, 2> _frame_submitted{};
    uint32_t _current_frame{0};
    uint64_t _epoch{0};
    uint64_t _submitted_epoch{0};
    uint64_t _completed_epoch{0};
    uint64_t _discarded_epoch{0};
    uint64_t _discard_seq{0};
    bool _recording{false};
    bool _debug_utils_enabled{false};
    std::string _debug_name;
};

} // namespace pictor
