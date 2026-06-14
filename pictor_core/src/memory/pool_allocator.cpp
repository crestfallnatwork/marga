#include "pictor_core/memory/pool_allocator.hpp"
#include "pictor_core/context/context.hpp"

namespace pictor {
PoolAllocator::PoolAllocator(Context *ctx) : _ctx(ctx) {
  vma::AllocationCreateInfo queryInfo{};
  queryInfo.usage = vma::MemoryUsage::eUnknown;
  queryInfo.requiredFlags = vk::MemoryPropertyFlagBits::eDeviceLocal;

  uint32_t memTypeIndex = 0;
  auto res = _ctx->vma_allocator.findMemoryTypeIndex(UINT32_MAX, queryInfo);

  if (!res.has_value()) {
      /* we need to refactor this */
      std::terminate();
  }

  vma::PoolCreateInfo poolInfo{};
  poolInfo.memoryTypeIndex = memTypeIndex;
  _pool = _ctx->vma_allocator.createPool(poolInfo).value;
}

PoolAllocator::~PoolAllocator() {
  if (_pool && _ctx) {
    _ctx->vma_allocator.destroyPool(_pool);
  }
}

AllocationInfo PoolAllocator::allocate(size_t bytes, size_t alignment) {
  PoolAllocatorParams params;
  params.size = bytes;
  params.type = AllocationType::eBuffer;

  params.bufferUsage = vk::BufferUsageFlagBits::eTransferSrc |
                       vk::BufferUsageFlagBits::eTransferDst |
                       vk::BufferUsageFlagBits::eStorageBuffer |
                       vk::BufferUsageFlagBits::eShaderDeviceAddress;

  return allocate(params);
}

AllocationInfo PoolAllocator::allocate(PoolAllocatorParams const &params) {
  AllocationInfo result{};
  result.type = params.type;

  vma::AllocationCreateInfo localAllocInfo{};
  localAllocInfo.pool = _pool;
  localAllocInfo.usage = vma::MemoryUsage::eUnknown;

  vk::BufferCreateInfo bufferInfo{};
  bufferInfo.size = params.size;
  bufferInfo.usage = params.bufferUsage;

  auto res = _ctx->vma_allocator.createBuffer(bufferInfo, localAllocInfo);
  result.res = res.result;
  result.buffer = res.value.second;
  result.allocation = res.value.first;

  if (result.allocation) {
    vma::AllocationInfo vmaInfo = _ctx->vma_allocator.getAllocationInfo(result.allocation);
    result.size = vmaInfo.size;
    result.offset = 0;
    result.mappedPtr = vmaInfo.pMappedData;

    vk::BufferDeviceAddressInfo bdaInfo{result.buffer};
    result.address = _ctx->device.getBufferAddress(bdaInfo);
  }

  return result;
}

void PoolAllocator::deallocate(AllocationInfo info) {
  _ctx->vma_allocator.destroyBuffer(info.buffer, info.allocation);
}

void PoolAllocator::flush(AllocationInfo const &info) {
  (void)_ctx->vma_allocator.flushAllocation(info.allocation, 0, VK_WHOLE_SIZE);
}

void PoolAllocator::invalidate(AllocationInfo const &info) {
  (void)_ctx->vma_allocator.invalidateAllocation(info.allocation, 0, VK_WHOLE_SIZE);
}

} // namespace pictor
