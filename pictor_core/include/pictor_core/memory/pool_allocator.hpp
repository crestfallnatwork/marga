/* SPDX-License-Identifier: MPL-2.0 */
/* Author: Himanshu Gupta <himanshu3@iisc.ac.in> */

#pragma once
#include "allocator.hpp"

namespace pictor {

class Context;

struct PoolAllocatorParams {
    size_t size;
    AllocationType type;
    vk::BufferUsageFlags bufferUsage;
};

class PoolAllocator : public Allocator {
    Context* _ctx;
    vma::Pool _pool;

  public:
    PoolAllocator(Context* ctx);
    ~PoolAllocator() override;

    AllocationInfo allocate(size_t bytes, size_t alignment) override;

    AllocationInfo allocate(PoolAllocatorParams const& params);

    void deallocate(AllocationInfo& info) override;

    void flush(AllocationInfo const& info) override;
    void invalidate(AllocationInfo const& info) override;
};
} // namespace pictor
