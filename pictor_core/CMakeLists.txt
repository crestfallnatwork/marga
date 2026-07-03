cmake_minimum_required(VERSION 3.15)
project(pictor-core CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Vulkan REQUIRED)

include(FetchContent)
FetchContent_Declare(
    vmahpp
    URL https://github.com/YaaZ/VulkanMemoryAllocator-Hpp/releases/download/v3.3.0%2B3/VulkanMemoryAllocator-Hpp-3.3.0.tar.gz
    URL_HASH SHA256=8c2a0573babe1f86f3241d12b4c9df40bb944649b7f4239c1c611797f8af6cd5
)

FetchContent_Declare(
    volk
    GIT_REPOSITORY https://github.com/zeux/volk.git
    GIT_TAG        1.4.350
)

FetchContent_Declare(
  slang
  URL https://github.com/shader-slang/slang/releases/download/v2026.10.2/slang-2026.10.2-linux-x86_64-glibc-2.27.tar.gz
  URL_HASH SHA256=035aaeaaadd6024d64da28d665685f793d0928831bf32cb2c37cb3e58c5ac707
)

FetchContent_MakeAvailable(vmahpp volk slang)
list(APPEND CMAKE_PREFIX_PATH "${slang_SOURCE_DIR}")

find_package(slang REQUIRED CONFIG)

add_library(pictor_core
    src/context/context.cpp
    src/execution/stream.cpp
    src/execution/shader_compiler.cpp
    src/memory/pool_allocator.cpp
    src/memory/pinned_memory_allocator.cpp
    src/memory/heap_allocator.cpp
)

target_include_directories(pictor_core
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_compile_definitions(pictor_core 
    PUBLIC 
        VULKAN_HPP_NO_EXCEPTIONS
        VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
        VK_NO_PROTOTYPES
)

target_link_libraries(pictor_core 
    PUBLIC 
        volk
        VulkanMemoryAllocator-Hpp::VulkanMemoryAllocator-Hpp
        slang::slang 
)

set(PICTOR_CORE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE INTERNAL "")
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/PictorCoreUtils.cmake)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    add_executable(library_test test/test.cpp)
    target_link_libraries(library_test 
        PRIVATE 
            pictor_core
    )
endif()
