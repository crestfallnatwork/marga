add_executable(pictor_bench_scan_vulkan
    scan_vulkan.cpp
)

target_link_libraries(pictor_bench_scan_vulkan
    PRIVATE
        pictor_algorithms
)

add_executable(pictor_bench_sort_vulkan
    sort_vulkan.cpp
)

target_link_libraries(pictor_bench_sort_vulkan
    PRIVATE
        pictor_algorithms
)

if(CUDAToolkit_FOUND)
    add_executable(pictor_bench_scan_cuda
        scan_cuda.cu
    )

    add_executable(pictor_bench_sort_cuda
        sort_cuda.cu
    )

    find_path(PICTOR_CUB_INCLUDE cub/device/device_scan.cuh
        HINTS
            "$ENV{CONDA_PREFIX}/include/rapids"
            "$ENV{CONDA_PREFIX}/targets/x86_64-linux/include/cccl"
            "$ENV{CONDA_PREFIX}/include"
            "/usr/local/cuda/include"
    )
    if(PICTOR_CUB_INCLUDE)
        target_include_directories(pictor_bench_scan_cuda SYSTEM PRIVATE ${PICTOR_CUB_INCLUDE})
        target_include_directories(pictor_bench_sort_cuda SYSTEM PRIVATE ${PICTOR_CUB_INCLUDE})
    endif()

    target_link_libraries(pictor_bench_scan_cuda
        PRIVATE
            CUDA::cudart
    )

    target_link_libraries(pictor_bench_sort_cuda
        PRIVATE
            CUDA::cudart
    )
endif()
