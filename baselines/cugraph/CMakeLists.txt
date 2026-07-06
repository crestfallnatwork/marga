cmake_minimum_required(VERSION 3.23)
project(cugraph_app LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

find_package(CUDAToolkit REQUIRED)
find_package(rmm REQUIRED)
find_package(raft REQUIRED)
find_package(cugraph REQUIRED)

find_package(OpenMP REQUIRED)

add_library(csr csr.cpp)

add_executable(bfs bfs.cu)
target_compile_options(bfs PRIVATE "-O3")
target_link_libraries(bfs
    PRIVATE 
        cugraph::cugraph 
        rmm::rmm 
        raft::raft
        OpenMP::OpenMP_CXX
        csr
)

add_executable(pagerank pagerank.cu)
target_compile_options(pagerank PRIVATE "-O3")
target_link_libraries(pagerank
    PRIVATE 
        cugraph::cugraph 
        rmm::rmm 
        raft::raft
        OpenMP::OpenMP_CXX
        csr
)

add_executable(tc tc.cu)
target_compile_options(tc PRIVATE "-O3" "-fopenmp")
target_link_libraries(tc
    PRIVATE 
        cugraph::cugraph 
        rmm::rmm 
        raft::raft
        OpenMP::OpenMP_CXX
        csr
)
add_executable(flat_pagerank flat_pagerank.cu)
target_compile_options(flat_pagerank PRIVATE "-O3")
target_link_libraries(flat_pagerank PRIVATE cugraph::cugraph rmm::rmm raft::raft OpenMP::OpenMP_CXX csr)
