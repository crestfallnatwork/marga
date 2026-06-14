# pictor-core

## Project Overview
`pictor-core` is a C++20 library providing a Vulkan memory management abstraction layer. It relies on the Vulkan Memory Allocator (`VulkanMemoryAllocator-Hpp`), which is fetched automatically via CMake. The library defines core abstractions like an `Allocator` interface, along with implementations like `PoolAllocator` and a high-level `memory_manager` for pooling device memory and workspace allocations. It wraps Vulkan handles (`vk::Buffer`, `vk::Image`) and VMA allocations (`vma::Allocation`) into a unified `AllocationInfo` structure.

## Main Technologies
*   **C++20**
*   **CMake** (Build system)
*   **Vulkan** (Graphics/Compute API)
*   **VulkanMemoryAllocator-Hpp** (Memory management backend)

## Directory Structure
*   `include/pictor_core/`: Contains the public header files.
    *   `memory/`: Defines memory allocation interfaces (`allocator.hpp`), types (`allocation_type.hpp`), and implementations (`pool_allocator.hpp`, `heap_allocator.hpp`).
*   `src/`: Contains the implementation files and tests.
    *   `memory_manager.cpp`: Implementation of a high-level memory manager that splits limits into data and workspace pools.
    *   `memory_manager_test.cpp` / `pool_allocator_test.cpp`: Test executables.

## Building and Running
The project uses standard CMake commands.

**To build the project:**
```bash
mkdir -p build
cd build
cmake ..
make
```

**To run the tests:**
```bash
# From the build directory
./memory_manager_test
```

## Development Conventions
*   **Namespaces:** The project primarily uses the `pictor` namespace (note: there are a few references to a `rasterdf` namespace in source files, which appear to be copy-paste artifacts).
*   **Vulkan C++ Bindings:** Uses `vulkan.hpp` conventions (e.g., `vk::Buffer`, `vk::Result`) and `vma::` for the C++ bindings of the Vulkan Memory Allocator.
*   **Exceptions:** Vulkan C++ exceptions are disabled via `VULKAN_HPP_NO_EXCEPTIONS` in CMake.
