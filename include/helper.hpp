#pragma once
#include <vulkan/vulkan.hpp>

#include "pictor_core/memory/allocation_type.hpp"

#include <iostream>



inline void CheckAllocation(pictor::AllocationInfo &a,
                            std::string to_print = "") {
  if (a.res != vk::Result::eSuccess) {
    std::cerr << std::format("Allocation Error: {}\n{}", vk::to_string(a.res),
                             to_print);
    std::terminate();
  }
}
