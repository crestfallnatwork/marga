# SPDX-License-Identifier: GPL-2.0-only
# Author: Himanshu Gupta <himanshu3@iisc.ac.in>

cmake_minimum_required(VERSION 3.20)
project(pictor_graphs)

set(CMAKE_CXX_STANDARD 20)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_definitions(PICTOR_DEBUG)
endif()

find_package(OpenMP REQUIRED)

add_subdirectory(pictor_core)

include_directories(include)
include_directories(external)

add_library(pictor_graphs src/csr.cpp)

target_link_libraries(pictor_graphs pictor_core OpenMP::OpenMP_CXX)

pictor_add_shader(bfs_shader src/shaders/bfs.slang)
add_library(pictor_graphs_bfs src/bfs.cpp src/bfs_validation.cpp)
target_link_libraries(pictor_graphs_bfs bfs_shader)

pictor_add_shader(pr_shader src/shaders/pr.slang)
add_library(pictor_graphs_pr src/pagerank.cpp)
target_link_libraries(pictor_graphs_pr pr_shader)

pictor_add_shader(tc_shader src/shaders/tc.slang)
add_library(pictor_graphs_tc src/tc.cpp)
target_link_libraries(pictor_graphs_tc tc_shader)

add_executable(bfs src/bfs_main.cpp)
target_link_libraries(bfs pictor_graphs pictor_core pictor_graphs_bfs)

add_executable(pagerank src/pagerank_main.cpp)
target_link_libraries(pagerank pictor_graphs pictor_core pictor_graphs_pr)

add_executable(tc src/tc_main.cpp)
target_link_libraries(tc pictor_graphs pictor_core pictor_graphs_tc)

add_executable(gen_vertices src/gen_vertices.cpp)
target_link_libraries(gen_vertices pictor_graphs pictor_core)
