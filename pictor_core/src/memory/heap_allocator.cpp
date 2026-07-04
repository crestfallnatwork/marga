/* SPDX-License-Identifier: MPL-2.0 */
/* Author: Himanshu Gupta <himanshu3@iisc.ac.in> */

#include "pictor_core/memory/heap_allocator.hpp"
#include "pictor_core/context/context.hpp"

namespace pictor {

Expected<std::unique_ptr<HeapAllocator>, vk::Result>
HeapAllocator::create(Context* ctx, Allocator* underlying, size_t max_heap_size)
{
    std::unique_ptr<HeapAllocator> heap_allocator{new HeapAllocator};
    heap_allocator->_ctx                  = ctx;
    heap_allocator->_underlying_allocator = underlying;
    AllocationInfo underlying_allocation =
        heap_allocator->_underlying_allocator->allocate(max_heap_size, 0);
    if (underlying_allocation.res != vk::Result::eSuccess) {
        return underlying_allocation.res;
    }
    heap_allocator->_underlying_allocation = underlying_allocation;
    ctx->fill(underlying_allocation, 0);
    ctx->fsync();
    vma::VirtualBlockCreateInfo vb_cr_info;
    vb_cr_info.size        = max_heap_size;
    heap_allocator->_block = vma::createVirtualBlock(vb_cr_info).value;
    return heap_allocator;
}

HeapAllocator::~HeapAllocator()
{
    _ctx->deallocate(_underlying_allocation);
    if (_block){
        _block.clearVirtualBlock();
        _block.destroy();
    }
}

AllocationInfo HeapAllocator::allocate(size_t bytes, size_t alignment)
{
    if(alignment == 0) alignment = 1;
    vma::VirtualAllocationCreateInfo info;
    alignment = (alignment + 15) / 16; /*align the alignment to 16 byte boundary*/
    info.setSize(bytes);
    info.setAlignment(alignment);
    size_t offset;
    auto v_alloc = _block.virtualAllocate(info, offset);
    auto alloc   = _underlying_allocation;
    if (v_alloc.result != vk::Result::eSuccess) {
        alloc.res = v_alloc.result;
        return alloc;
    }
    alloc.alloc = this;
    alloc.vAlloc = v_alloc.value;
    alloc.size   = bytes;
    alloc.offset = offset;
    if (alloc.address)
        alloc.address += offset;
    if (alloc.mappedPtr) {
        alloc.mappedPtr =
            reinterpret_cast<void*>(reinterpret_cast<size_t>(alloc.mappedPtr) + offset);
    }
    return alloc;
}

AllocationInfo HeapAllocator::allocate(HeapAllocatorParams const& params)
{
    return allocate(params.size, params.alignment);
}

void HeapAllocator::deallocate(AllocationInfo& info)
{
    if (info.vAlloc)
        _block.free(info.vAlloc);
}

void HeapAllocator::flush(AllocationInfo const& info)
{
    (void)_ctx->vma_allocator.flushAllocation(_underlying_allocation.allocation, info.offset,
                                              info.size);
}

void HeapAllocator::invalidate(AllocationInfo const& info)
{
    (void)_ctx->vma_allocator.invalidateAllocation(_underlying_allocation.allocation, info.offset,
                                                   info.size);
}

} // namespace pictor
