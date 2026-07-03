/* SPDX-License-Identifier: MPL-2.0 */
/* Author: Himanshu Gupta <himanshu3@iisc.ac.in> */

#pragma once

#include "allocator.hpp"
#include "pictor_core/context/result_type.hpp"

#include <memory>

namespace pictor {

class Context;

struct HeapAllocatorParams {
    size_t size;
    size_t alignment;
};

class HeapAllocator : public Allocator {
    Context* _ctx;
    Allocator* _underlying_allocator;
    AllocationInfo _underlying_allocation;
    vma::VirtualBlock _block;
    HeapAllocator() = default;

  public:
    static Expected<std::unique_ptr<HeapAllocator>, vk::Result>
    create(Context* ctx, Allocator* underlying, size_t max_heap_size);
    virtual ~HeapAllocator() override;

    virtual AllocationInfo allocate(size_t bytes, size_t alignment) override;

    AllocationInfo allocate(HeapAllocatorParams const& params);

    virtual void deallocate(AllocationInfo& info) override;

    virtual void flush(AllocationInfo const& info) override;
    virtual void invalidate(AllocationInfo const& info) override;
};

} // namespace pictor
