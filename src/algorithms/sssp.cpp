#include "sssp.hpp"
#include <stdexcept>

std::vector<float> run_sssp(OptixPipeline pipe,
    const OptixShaderBindingTable& sbt,
    CUdeviceptr d_params,
    const Params& baseParams,
    int source,
    int num_vertices,
    GPUMemoryManager& mm,
    const std::vector<uint32_t>& uasp_first,
    CUstream streamOptix) {
    throw std::runtime_error("run_sssp not implemented");
}

void run_sssp_hybrid(OptixPipeline pipe,
    const OptixShaderBindingTable& sbt,
    CUdeviceptr d_params,
    const Params& baseParams,
    int source,
    int num_vertices,
    GPUMemoryManager& mm,
    const std::vector<uint32_t>& uasp_first,
    CUstream stream,
    uint32_t deg) {
    throw std::runtime_error("run_sssp_hybrid not implemented");
}
