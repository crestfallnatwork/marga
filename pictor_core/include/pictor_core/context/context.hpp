/* SPDX-License-Identifier: MPL-2.0 */
/* Author: Himanshu Gupta <himanshu3@iisc.ac.in> */

#pragma once

#include "pictor_core/context/device_info.hpp"
#include "pictor_core/context/result_type.hpp"
#include "pictor_core/execution/shader_compiler.hpp"
#include "pictor_core/execution/stream.hpp"
#include "pictor_core/memory/allocation_type.hpp"
#include "pictor_core/memory/pinned_memory_allocator.hpp"
#include "pictor_core/memory/pool_allocator.hpp"

#include <vulkan/vulkan.hpp>

#include <vk_mem_alloc.hpp>

#include <limits>
#include <memory>
#include <string>
#include <string_view>

namespace pictor {

class Context {
  public:
    static Expected<std::unique_ptr<Context>, vk::Result> create();

    ~Context();

    Context(const Context&)            = delete;
    Context& operator=(const Context&) = delete;

    StreamHandle create_stream();
    vk::Result submit(StreamHandle handle = cDefaultStream);
    vk::Result fsync(StreamHandle handle = cDefaultStream);
    Stream* get_stream(StreamHandle handle);

    void wait_idle() const
    {
        if (device.waitIdle() != vk::Result::eSuccess)
            return;
        for (const auto& stream : _streams) {
            stream->_completed_epoch = stream->_submitted_epoch;
        }
    }
    void sync(StreamHandle handle = cDefaultStream);
    void barrier(const AllocationInfo& resource, Access from, Access to,
                 StreamHandle handle = cDefaultStream);
    void barrier(const std::vector<ResourceBarrier>& barriers,
                 StreamHandle handle = cDefaultStream);

    void copy(const AllocationInfo& src, const AllocationInfo& dst,
              StreamHandle stream = cDefaultStream);
    void fill(const AllocationInfo& dst, uint32_t value, size_t offset = 0,
              size_t size         = std::numeric_limits<size_t>::max(),
              StreamHandle stream = cDefaultStream);
    void fill(const AllocationInfo& dst, size_t offset, size_t size, uint32_t value,
              StreamHandle stream = cDefaultStream);
    void update(const AllocationInfo& dst, size_t offset, size_t size, const void* data,
                StreamHandle stream = cDefaultStream);

    void invalidate(AllocationInfo& alloc);
    void flush(AllocationInfo& alloc);
    void deallocate(AllocationInfo& alloc);

    [[nodiscard]] size_t get_device_free_memory() const;
    [[nodiscard]] size_t get_host_free_memory() const;
    void set_host_allocator(std::unique_ptr<Allocator>&& allocator);
    void set_device_allocator(std::unique_ptr<Allocator>&& allocator);

    [[nodiscard]] bool debug_utils_enabled() const
    {
        return _debug_utils_enabled;
    }

    template <typename TObject> void set_name(const TObject& object, std::string_view name)
    {
        if (!_debug_utils_enabled || name.empty()) {
            return;
        }
        (void)device.setDebugUtilsObjectNameEXT(object, std::string(name));
    }

    vk::Instance instance;
    vk::PhysicalDevice physical_device;
    vk::Device device;

    vk::Queue queue;
    uint32_t queue_family_index{0};

    vma::Allocator vma_allocator;
    uint32_t subgroup_size{32};
    DeviceInfo device_info;

    Allocator* pinned_allocator;
    Allocator* pool_allocator;

    std::unique_ptr<PinnedMemoryAllocator> root_pinned_allocator;
    std::unique_ptr<PoolAllocator> root_pool_allocator;

    std::unique_ptr<ShaderCompiler> shader_compiler;

  private:
    Context() = default;

    vk::Result _init();
    vk::Result _create_instance();
    vk::Result _select_physical_device();
    vk::Result _gather_device_info();
    vk::Result _create_logical_device();
    vk::Result _setup_vma();
    vk::Result _setup_allocators();
    vk::Result _setup_compiler();
    vk::Result _setup_streams();

    std::vector<std::unique_ptr<Stream>> _streams;
    bool _debug_utils_enabled{false};
    bool _shader_debug_info{false};
};

} // namespace pictor
