#pragma once

#include "pictor_core/execution/synchronization.hpp"
#include "pictor_core/memory/allocation_type.hpp"
#include <array>
#include <vulkan/vulkan.hpp>

namespace pictor {

class Context;

using StreamHandle = uint32_t;
inline constexpr StreamHandle cDefaultStream = 0;

using RecordCallback = void (*)(vk::CommandBuffer cmd, void *user_data);

class Stream {
public:
  Stream(vk::Device device, uint32_t queue_family_index);
  ~Stream();

  Stream(const Stream &) = delete;
  Stream &operator=(const Stream &) = delete;

  vk::Result begin();
  vk::Result end();
  vk::Result flush(vk::Queue queue);

  // Wait for all work in this stream to finish
  void synchronize();

  // Resource-centric barrier
  void barrier(const AllocationInfo &resource, Access from, Access to);

  void record(RecordCallback cb, void *user_data);

  vk::Device device;
  vk::CommandPool command_pool;

  // The buffer currently being recorded (if any)
  vk::CommandBuffer current_command_buffer() const {
    return _command_buffers[_current_frame];
  }

private:
  friend class Context;

  void _copy(const AllocationInfo &src, const AllocationInfo &dst);

  std::array<vk::CommandBuffer, 2> _command_buffers;
  std::array<vk::Fence, 2> _fences;
  uint32_t _current_frame{0};
  bool _recording{false};
};

} // namespace pictor
