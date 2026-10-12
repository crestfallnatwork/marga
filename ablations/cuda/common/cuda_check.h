#pragma once

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

#define CUDA_CHECK(expr)                                                           \
    do {                                                                           \
        cudaError_t err_ = (expr);                                                 \
        if (err_ != cudaSuccess) {                                                 \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorName(err_), \
                         __FILE__, __LINE__, cudaGetErrorString(err_));                \
            std::exit(1);                                                          \
        }                                                                          \
    } while (0)

// Kernel launches have no return value; check the launch immediately so that
// configuration errors (wrong arch, bad grid, ...) never fail silently.
#define CUDA_CHECK_LAUNCH() CUDA_CHECK(cudaGetLastError())
