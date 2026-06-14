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
  Context *_ctx;

public:
  PinnedMemoryAllocator(Context *ctx);
  virtual ~PinnedMemoryAllocator() override;

  virtual AllocationInfo allocate(size_t bytes, size_t alignment) override;

  AllocationInfo allocate(PinnedMemoryAllocatorParams const &params);

  virtual void deallocate(AllocationInfo &info) override;

  virtual void flush(AllocationInfo const &info) override;
  virtual void invalidate(AllocationInfo const &info) override;
};

} // namespace pictor
