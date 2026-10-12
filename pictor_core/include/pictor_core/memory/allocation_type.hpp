/* SPDX-License-Identifier: MPL-2.0 */
/* Author: Himanshu Gupta <himanshu3@iisc.ac.in> */

#pragma once

#include <cstdint>

#include <vk_mem_alloc.hpp>

namespace pictor {

enum class AllocationType : uint8_t {
    eBuffer               = 0,
    eImage                = 1,
    eVirtualSuballocation = 2
};

struct Allocator;
struct AllocationInfo {
    vk::Result res;
    vk::Buffer buffer{VK_NULL_HANDLE};
    union {
        vma::Allocation allocation{VK_NULL_HANDLE};
        vma::VirtualAllocation vAlloc;
    };

    void* mappedPtr{nullptr};
    size_t size{0};
    size_t offset{0};
    AllocationType type{AllocationType::eBuffer};

    VkDeviceAddress address{0};

    Allocator* alloc = nullptr;

    // Explicit default constructor: the anonymous union has a variant member
    // (vma::VirtualAllocation) with a non-trivial default constructor, which
    // would otherwise implicitly delete this constructor on GCC.
    AllocationInfo() : allocation{VK_NULL_HANDLE} {}

    template <typename T>
    operator T*() const{
        return reinterpret_cast<T*>(address);
    }
};
} // namespace pictor
