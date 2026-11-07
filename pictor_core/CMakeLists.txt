cmake_minimum_required(VERSION 3.15)
project(pictor-core CXX)

# pictor_add_shader emits generated headers as INTERFACE sources consumed from other directories
# (e.g. algorithms/ -> bench/). CMP0118 makes the GENERATED property visible in all directories,
# without which a newly added shader is not found at configure time.
if(POLICY CMP0118)
    cmake_policy(SET CMP0118 NEW)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
enable_testing()

option(PICTOR_ENABLE_DEBUG_UTILS
    "Enable VK_EXT_debug_utils object names and labels by default (runtime override: PICTOR_DEBUG_UTILS)"
    ON)

option(PICTOR_ENABLE_SHADER_DEBUG_INFO
    "Compile Slang shaders with NonSemantic debug info by default (runtime override: PICTOR_SHADER_DEBUG_INFO)"
    OFF)

option(PICTOR_ENABLE_NGFX_SDK
    "Compile in the Nsight Graphics SDK one-shot GPU Trace hooks (runtime gate: PICTOR_NGFX_SDK=1)"
    OFF)

set(PICTOR_NGFX_SDK_INCLUDE_DIR
    "/opt/nsight-graphics/latest/SDKs/NsightGraphicsSDK/0.9.0/include"
    CACHE PATH "Nsight Graphics SDK include directory")

option(PICTOR_BUILD_BENCHMARKS "Build the comparison benchmark harnesses (not tests)" OFF)

find_package(Vulkan REQUIRED)

include(FetchContent)

# VMA-Hpp 3.3.0 needs a recent Vulkan-Hpp (e.g. vk::exchange, which distro packages such as
# Ubuntu 22.04's 1.3.283 lack). Pin the headers instead of relying on the system ones.
# Populate only (no add_subdirectory) to avoid colliding with FindVulkan's Vulkan::Headers.
FetchContent_Declare(
    vulkan_headers
    GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git
    GIT_TAG        v1.4.350
)
FetchContent_GetProperties(vulkan_headers)
if(NOT vulkan_headers_POPULATED)
    FetchContent_Populate(vulkan_headers)
endif()

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

if(WIN32)
    FetchContent_Declare(
        slang
        URL https://github.com/shader-slang/slang/releases/download/v2026.10.2/slang-2026.10.2-windows-x86_64.zip
        URL_HASH SHA256=f21fca4ba78bfb366ef3b282b4926e3efca61c2ffd72a482b024c9f8413331d0
    )
else()
    FetchContent_Declare(
        slang
        URL https://github.com/shader-slang/slang/releases/download/v2026.10.2/slang-2026.10.2-linux-x86_64-glibc-2.27.tar.gz
        URL_HASH SHA256=035aaeaaadd6024d64da28d665685f793d0928831bf32cb2c37cb3e58c5ac707
    )
endif()

FetchContent_MakeAvailable(vmahpp volk slang)
list(APPEND CMAKE_PREFIX_PATH "${slang_SOURCE_DIR}")

# volk compiles volk.c against the Vulkan API headers; use the pinned headers so it does not
# depend on whatever the system provides.
if(TARGET volk)
    target_include_directories(volk BEFORE PUBLIC
        $<BUILD_INTERFACE:${vulkan_headers_SOURCE_DIR}/include>)
endif()

find_package(slang REQUIRED CONFIG)

add_executable(pictor_reflect tools/pictor_reflect/main.cpp)
target_link_libraries(pictor_reflect PRIVATE slang::slang)

add_library(pictor_core
    src/context/context.cpp
    src/execution/stream.cpp
    src/execution/gpu_timer.cpp
    src/execution/shader_compiler.cpp
    src/memory/pool_allocator.cpp
    src/memory/pinned_memory_allocator.cpp
    src/memory/heap_allocator.cpp
    src/profiling/gputrace.cpp
)

target_include_directories(pictor_core
    BEFORE PUBLIC
        $<BUILD_INTERFACE:${vulkan_headers_SOURCE_DIR}/include>
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

if(PICTOR_ENABLE_DEBUG_UTILS)
    target_compile_definitions(pictor_core PUBLIC PICTOR_DEBUG_UTILS=1)
else()
    target_compile_definitions(pictor_core PUBLIC PICTOR_DEBUG_UTILS=0)
endif()

if(PICTOR_ENABLE_SHADER_DEBUG_INFO)
    target_compile_definitions(pictor_core PUBLIC PICTOR_SHADER_DEBUG_INFO=1)
else()
    target_compile_definitions(pictor_core PUBLIC PICTOR_SHADER_DEBUG_INFO=0)
endif()

if(PICTOR_ENABLE_NGFX_SDK)
    target_compile_definitions(pictor_core PRIVATE PICTOR_NGFX_SDK=1)
    target_include_directories(pictor_core SYSTEM PRIVATE ${PICTOR_NGFX_SDK_INCLUDE_DIR})
    target_link_libraries(pictor_core PRIVATE ${CMAKE_DL_LIBS})
endif()

target_link_libraries(pictor_core
    PUBLIC
        volk
        VulkanMemoryAllocator-Hpp::VulkanMemoryAllocator-Hpp
        slang::slang
)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/PictorCoreUtils.cmake)

add_subdirectory(algorithms)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    add_executable(library_test test/test.cpp)
    target_link_libraries(library_test
        PRIVATE
            pictor_core
    )
    add_test(NAME library_test COMMAND library_test)

    add_executable(device_info_test test/device_info_test.cpp)
    target_link_libraries(device_info_test
        PRIVATE
            pictor_core
    )
    add_test(NAME device_info COMMAND device_info_test)

    add_executable(gpu_timer_test test/gpu_timer_test.cpp)
    target_link_libraries(gpu_timer_test
        PRIVATE
            pictor_core
    )
    add_test(NAME gpu_timer COMMAND gpu_timer_test)

    pictor_add_shader(module_test_shader test/shaders/module_test.slang INCLUDE_DIRS test/modules)
    add_executable(module_test test/module_test.cpp)
    target_link_libraries(module_test
        PRIVATE
            pictor_core
            module_test_shader
    )
    add_test(NAME shader_module COMMAND module_test)

    pictor_add_shader(tunable_test_shader test/shaders/tunable_test.slang INCLUDE_DIRS test/modules)
    add_executable(tunable_test test/tunable_test.cpp)
    target_link_libraries(tunable_test
        PRIVATE
            pictor_core
            tunable_test_shader
    )
    add_test(NAME shader_tunables COMMAND tunable_test)

    # Both translation units include the generated header and ODR-use defines(); the link
    # proves the generated definition is inline (a non-inline body collides here).
    add_executable(tunable_multi_tu test/tunables_multi_tu_a.cpp test/tunables_multi_tu_b.cpp)
    target_link_libraries(tunable_multi_tu
        PRIVATE
            pictor_core
            tunable_test_shader
    )
    add_test(NAME shader_tunables_multi_tu COMMAND tunable_multi_tu)

    pictor_add_shader(tuner_test_shader test/shaders/tuner_test.slang)
    add_executable(tuner_test test/tuner_test.cpp)
    target_link_libraries(tuner_test
        PRIVATE
            pictor_core
            tuner_test_shader
    )
    add_test(NAME shader_tuner COMMAND tuner_test)

    pictor_add_shader(multi_push_test_shader test/shaders/multi_push.slang)
    pictor_add_shader(same_type_push_shader test/shaders/same_type_push.slang)
    add_executable(multi_push_test test/multi_push_test.cpp)
    target_link_libraries(multi_push_test
        PRIVATE
            pictor_core
            multi_push_test_shader
            same_type_push_shader
    )
    add_test(NAME multi_push COMMAND multi_push_test)

    set(PICTOR_REFLECT_TEST_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test/expect_reflect_result.cmake)

    function(pictor_expect_reflect_failure name shader expected)
        add_test(NAME ${name}
                 COMMAND ${CMAKE_COMMAND}
                         -DREFLECTOR=$<TARGET_FILE:pictor_reflect>
                         -DSHADER=${CMAKE_CURRENT_SOURCE_DIR}/test/shaders/${shader}
                         -DOUTDIR=${CMAKE_CURRENT_BINARY_DIR}/reflect_tests/${name}
                         -DEXPECT_FAILURE=ON
                         "-DEXPECTED=${expected}"
                         -P ${PICTOR_REFLECT_TEST_SCRIPT})
    endfunction()

    function(pictor_expect_reflect_output name shader expected)
        if(ARGN)
            string(JOIN "|" absent ${ARGN})
        else()
            set(absent "")
        endif()
        add_test(NAME ${name}
                 COMMAND ${CMAKE_COMMAND}
                         -DREFLECTOR=$<TARGET_FILE:pictor_reflect>
                         -DSHADER=${CMAKE_CURRENT_SOURCE_DIR}/test/shaders/${shader}
                         -DOUTDIR=${CMAKE_CURRENT_BINARY_DIR}/reflect_tests/${name}
                         "-DEXPECTED=${expected}"
                         "-DEXPECTED_ABSENT=${absent}"
                         -P ${PICTOR_REFLECT_TEST_SCRIPT})
    endfunction()

    function(pictor_expect_reflect_header name shader expected)
        add_test(NAME ${name}
                 COMMAND ${CMAKE_COMMAND}
                         -DREFLECTOR=$<TARGET_FILE:pictor_reflect>
                         -DSHADER=${CMAKE_CURRENT_SOURCE_DIR}/test/shaders/${shader}
                         -DOUTDIR=${CMAKE_CURRENT_BINARY_DIR}/reflect_tests/${name}
                         -DEXPECTED=
                         "-DEXPECTED_HEADER=${expected}"
                         -P ${PICTOR_REFLECT_TEST_SCRIPT})
    endfunction()

    pictor_expect_reflect_failure(shader_tunable_reject bad_tunable.slang
                                  "PICTOR_BAD' has the non-literal default")
    pictor_expect_reflect_failure(shader_tunable_reject_empty bad_field_empty.slang
                                  "PICTOR_' maps to the invalid C++ field name ''")
    pictor_expect_reflect_failure(shader_tunable_reject_digit bad_field_digit.slang
                                  "PICTOR_1' maps to the invalid C++ field name '1'")
    pictor_expect_reflect_failure(shader_tunable_reject_keyword bad_field_keyword.slang
                                  "PICTOR_FOR' maps to the invalid C++ field name 'for'")
    pictor_expect_reflect_failure(shader_tunable_reject_collision bad_collision.slang
                                  "tunables 'PICTOR_FOO' and 'PICTOR_foo' both map to the field 'foo'")

    pictor_expect_reflect_output(shader_tunable_ghosts ghost_tunables.slang
                                 "tunables: 1|PICTOR_LIVE = 3" PICTOR_COMMENTED PICTOR_DEAD)
    pictor_expect_reflect_output(shader_tunable_bom bom_tunable.slang "tunables: 1|PICTOR_BOM = 5")

    pictor_expect_reflect_output(shader_multi_push multi_push.slang
                                 "entry delta_compute [compute] threadgroup 1x1x1|push constant: AlphaIn|push constant: BetaIn|push constant: GammaIn")
    pictor_expect_reflect_header(shader_multi_push_delta multi_push.slang
                                 "using DeltaComputeExecutor = pictor::ComputeExecutor<DeltaCompute, void>;")
    pictor_expect_reflect_failure(shader_multi_push_mismatch bad_multi_push.slang
                                  "must share one push-constant layout")
    pictor_expect_reflect_failure(shader_multi_push_two_blocks bad_two_blocks.slang
                                  "distinct push-constant layouts")
    pictor_expect_reflect_failure(shader_multi_push_imported import_push.slang
                                  "push constants must be declared in the entry file")
    pictor_expect_reflect_failure(shader_multi_push_collision collide_push.slang
                                  "not declared in the entry file")

    add_executable(shader_debug_test test/shader_debug_test.cpp)
    target_link_libraries(shader_debug_test
        PRIVATE
            pictor_core
    )
    add_test(NAME shader_debug_info COMMAND shader_debug_test)

    add_subdirectory(examples)
endif()

if(PICTOR_BUILD_BENCHMARKS)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES 75;80;89 CACHE STRING
                "CUDA architectures: dev sm_75, A100 sm_80, RTX 4090 sm_89")
        endif()
        set(CMAKE_CUDA_STANDARD 20)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        enable_language(CUDA)
    else()
        message(STATUS "CUDAToolkit not found; building the Vulkan benchmarks only")
    endif()
    add_subdirectory(bench)
endif()
