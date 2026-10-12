cmake_minimum_required(VERSION 3.24)
project(marga_cuda_ablations LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Target compute capability (89 = Ada, 75 = Turing, ...). nvcc defaults to
# sm_75 and CMake records that default in CMAKE_CUDA_ARCHITECTURES before this
# file runs, so guard on our own override variable instead of
# `DEFINED CMAKE_CUDA_ARCHITECTURES`. Override with -DMARGA_CUDA_ARCH=<cc>.
set(MARGA_CUDA_ARCH 89 CACHE STRING "CUDA compute capability to target")
set(CMAKE_CUDA_ARCHITECTURES ${MARGA_CUDA_ARCH} CACHE STRING "CUDA architectures" FORCE)

find_package(CUDAToolkit REQUIRED)

# Triangle-counting ablation: one thread per edge with MARGA's pruning
# (`--cull none|orient|bounded`), plus the `--preorient` orientation pre-pass.
#
# The harness shares the native host importer/cache and CPU validators with
# MARGA; they are compiled directly here so the ablation build stays free of
# the Vulkan dependency.
set(MARGA_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)

add_executable(tc_ablate
  tc/tc_main.cu
  tc/tc_runner.cu
  ${MARGA_ROOT}/src/host_graph.cpp
  ${MARGA_ROOT}/src/tc_validation.cpp
)
# CUB ships with the CUDA toolkit under the CCCL include root.
target_include_directories(tc_ablate PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${MARGA_ROOT}/include
  ${CUDAToolkit_INCLUDE_DIRS})
