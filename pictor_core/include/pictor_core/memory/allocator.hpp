/* SPDX-License-Identifier: MPL-2.0 */
/* Author: Himanshu Gupta <himanshu3@iisc.ac.in> */

#pragma once

#include "allocation_type.hpp"

namespace pictor {
class Allocator {
public:
  virtual ~Allocator() = default;

  virtual AllocationInfo allocate(size_t bytes, size_t alignment) = 0;

  virtual void deallocate(AllocationInfo &info) = 0;

  virtual void flush(AllocationInfo const &info) = 0;
  virtual void invalidate(AllocationInfo const &info) = 0;
};

} // namespace pictor
