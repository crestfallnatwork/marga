/* SPDX-License-Identifier: MPL-2.0 */
/* Author: Himanshu Gupta <himanshu3@iisc.ac.in> */

#pragma once

#include "allocator.hpp"

namespace pictor {

class Context;

struct PinnedMemoryAllocatorParams {
    size_t size;
    AllocationType type;
    vk::BufferUsageFlags bufferUsage;
};

class PinnedMemoryAllocator : public Allocator {
    Context* _ctx;

  public:
    PinnedMemoryAllocator(Context* ctx);
    ~PinnedMemoryAllocator() override;

    AllocationInfo allocate(size_t bytes, size_t alignment) override;

    AllocationInfo allocate(PinnedMemoryAllocatorParams const& params);

    void deallocate(AllocationInfo& info) override;

    void flush(AllocationInfo const& info) override;
    void invalidate(AllocationInfo const& info) override;
};

} // namespace pictor
