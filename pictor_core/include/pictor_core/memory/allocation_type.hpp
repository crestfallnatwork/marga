#pragma once

#include <cstdint>

#include <vk_mem_alloc.hpp>

namespace pictor {

enum class AllocationType : uint8_t {
  eBuffer = 0,
  eImage = 1,
  eVirtualSuballocation = 2
};

struct Allocator;
struct AllocationInfo {
  vk::Result res;
  union {
    vk::Buffer buffer{VK_NULL_HANDLE};
    vk::Image image;
  };
  union {
    vma::Allocation allocation{VK_NULL_HANDLE};
    vma::VirtualAllocation vAlloc;
  };

  void *mappedPtr{nullptr};
  size_t size{0};
  size_t offset{0};
  AllocationType type{AllocationType::eBuffer};

  union {
    VkDeviceAddress address{0};
    struct {
      VkImageView view;
      VkImageLayout layout;
    } img;
  };
  
  Allocator *alloc;
};
} // namespace pictor
