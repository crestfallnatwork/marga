#pragma once

#include <vma/vma.hpp>

#include <cstddef>
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>

#include <rasterdf/core/vk_constants.hpp>

namespace rasterdf {

enum class allocation_type : uint8_t {
    buffer                 = 0,
    image                  = 1,
    virtual_suballocation  = 2
};

struct allocation_info {
    union {
        VkBuffer buffer{VK_NULL_HANDLE};
        VkImage  image;
    };
    union {
        VmaAllocation        allocation{VK_NULL_HANDLE};
        VmaVirtualAllocation vAlloc;
    };

    void* mapped_ptr{nullptr};   
    size_t          size{0};
    size_t          offset{0};  
    allocation_type type{allocation_type::buffer};

    mutable VkPipelineStageFlags2 last_writing_stage{VK_PIPELINE_STAGE_2_NONE};
    mutable VkAccessFlags2        last_access_mask{VK_ACCESS_2_NONE};

    union {
        VkDeviceAddress address{0};  
        struct {
            VkImageView      view;
            VkImageLayout    layout;
        } img;
    };
};

class memory_resource {
 public:
  virtual ~memory_resource() = default;

  virtual allocation_info allocate(size_t bytes,
                                     size_t alignment) = 0;
  virtual allocation_info allocate(size_t bytes,
                                   VkBufferUsageFlags usage,
                                   VmaMemoryUsage memory_usage = VMA_MEMORY_USAGE_AUTO) = 0;

  virtual void deallocate(allocation_info info) = 0;

  virtual void flush(allocation_info const &info) = 0;
  virtual void invalidate(allocation_info const &info) = 0;

  virtual VmaAllocator allocator() const = 0;
};

}  // namespace rasterdf
