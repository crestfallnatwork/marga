/* SPDX-License-Identifier: MPL-2.0 */
/* Author: Himanshu Gupta <himanshu3@iisc.ac.in> */

#include "pictor_core/memory/pinned_memory_allocator.hpp"
#include "pictor_core/context/context.hpp"

namespace pictor {

PinnedMemoryAllocator::PinnedMemoryAllocator(Context* ctx) : _ctx(ctx) {}

PinnedMemoryAllocator::~PinnedMemoryAllocator() {}

AllocationInfo PinnedMemoryAllocator::allocate(size_t bytes, size_t alignment)
{
    PinnedMemoryAllocatorParams params;
    params.size = bytes;
    params.type = AllocationType::eBuffer;

    params.bufferUsage = vk::BufferUsageFlagBits::eTransferSrc |
                         vk::BufferUsageFlagBits::eTransferDst |
                         vk::BufferUsageFlagBits::eShaderDeviceAddress;

    return allocate(params);
}

AllocationInfo PinnedMemoryAllocator::allocate(PinnedMemoryAllocatorParams const& params)
{
    AllocationInfo result{};
    result.type = params.type;

    vma::AllocationCreateInfo localAllocInfo{};
    localAllocInfo.usage = vma::MemoryUsage::eAutoPreferHost;
    localAllocInfo.flags = vma::AllocationCreateFlagBits::eHostAccessSequentialWrite |
                           vma::AllocationCreateFlagBits::eMapped;

    vk::BufferCreateInfo bufferInfo{};
    bufferInfo.size  = params.size;
    bufferInfo.usage = params.bufferUsage;

    auto res          = _ctx->vma_allocator.createBuffer(bufferInfo, localAllocInfo);
    result.alloc      = this;
    result.res        = res.result;
    result.buffer     = res.value.second;
    result.allocation = res.value.first;

    if (result.allocation) {
        vma::AllocationInfo vmaInfo = _ctx->vma_allocator.getAllocationInfo(result.allocation);
        result.size                 = params.size;
        result.offset               = 0;
        result.mappedPtr            = vmaInfo.pMappedData;

        vk::BufferDeviceAddressInfo bdaInfo{result.buffer};
        result.address = _ctx->device.getBufferAddress(bdaInfo);
    }

    return result;
}

void PinnedMemoryAllocator::deallocate(AllocationInfo& info)
{
    _ctx->vma_allocator.destroyBuffer(info.buffer, info.allocation);
}

void PinnedMemoryAllocator::flush(AllocationInfo const& info)
{
    (void)_ctx->vma_allocator.flushAllocation(info.allocation, 0, VK_WHOLE_SIZE);
}

void PinnedMemoryAllocator::invalidate(AllocationInfo const& info)
{
    (void)_ctx->vma_allocator.invalidateAllocation(info.allocation, 0, VK_WHOLE_SIZE);
}

} // namespace pictor
