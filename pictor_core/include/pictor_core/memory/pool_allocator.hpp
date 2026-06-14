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
  Context *_ctx;
  vma::Pool _pool;

public:
  PoolAllocator(Context *ctx);
  virtual ~PoolAllocator() override;

  virtual AllocationInfo allocate(size_t bytes, size_t alignment) override;

  AllocationInfo allocate(PoolAllocatorParams const &params);

  virtual void deallocate(AllocationInfo &info) override;

  virtual void flush(AllocationInfo const &info) override;
  virtual void invalidate(AllocationInfo const &info) override;
};
} // namespace pictor
