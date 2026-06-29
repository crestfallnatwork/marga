/* SPDX-License-Identifier: MPL-2.0 */
/* Author: Himanshu Gupta <himanshu3@iisc.ac.in> */

#pragma once

#include "pictor_core/context/result_type.hpp"
#include "pictor_core/execution/shader_compiler.hpp"
#include "pictor_core/execution/stream.hpp"
#include "pictor_core/memory/allocation_type.hpp"
#include "pictor_core/memory/pinned_memory_allocator.hpp"
#include "pictor_core/memory/pool_allocator.hpp"

#include <vulkan/vulkan.hpp>

#include <vk_mem_alloc.hpp>

#include <memory>

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
        (void)device.waitIdle();
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

    vk::Instance instance;
    vk::PhysicalDevice physical_device;
    vk::Device device;

    vk::Queue queue;
    uint32_t queue_family_index{0};

    vma::Allocator vma_allocator;
    uint32_t subgroup_size{32};

    std::unique_ptr<PinnedMemoryAllocator> pinned_allocator;
    std::unique_ptr<PoolAllocator> pool_allocator;
    std::unique_ptr<ShaderCompiler> shader_compiler;

  private:
    Context() = default;

    vk::Result _init();
    vk::Result _create_instance();
    vk::Result _select_physical_device();
    vk::Result _create_logical_device();
    vk::Result _setup_vma();
    vk::Result _setup_allocators();
    vk::Result _setup_compiler();
    vk::Result _setup_streams();

    std::vector<std::unique_ptr<Stream>> _streams;
};

} // namespace pictor
