#include <cugraph/algorithms.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/device_span.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <iostream>
#include <chrono>
#include <tuple>
#include "csr.hpp"

template <typename vertex_t, typename edge_t, typename weight_t>
__global__ void compute_out_degrees_kernel(
    const vertex_t* __restrict__ indices,
    edge_t num_edges,
    weight_t* __restrict__ out_degrees) 
{
    edge_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_edges) {
        vertex_t src = indices[tid];
        atomicAdd(&out_degrees[src], (weight_t)1.0);
    }
}

template <typename F>
float time_gpu_ms(raft::handle_t const& handle, F&& fn) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    auto stream = handle.get_stream();
    cudaEventRecord(start, stream);
    fn();
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <graph.mtx>\n";
        return 1;
    }

    using vertex_t = int32_t;
    using edge_t = int32_t;
    using weight_t = float;
    using result_t = float;

    std::string filename = argv[1];
    
    // Import graph (CSC layout for PageRank, as cugraph requires store_transposed = true)
    auto h_graph = Graph<vertex_t, edge_t>::import(filename, false, GraphLayout::CSC);

    raft::handle_t handle;
    auto stream = handle.get_stream();

    std::cout << "Transferring graph to GPU...\n";
    rmm::device_uvector<edge_t> d_offsets(h_graph.row_offsets.size(), stream);
    rmm::device_uvector<vertex_t> d_indices(h_graph.column_indices.size(), stream);

    raft::update_device(d_offsets.data(), h_graph.row_offsets.data(), h_graph.row_offsets.size(), stream);
    raft::update_device(d_indices.data(), h_graph.column_indices.data(), h_graph.column_indices.size(), stream);
    handle.sync_stream();

    auto offsets_span = raft::device_span<edge_t const>(d_offsets.data(), d_offsets.size());
    auto indices_span = raft::device_span<vertex_t const>(d_indices.data(), d_indices.size());

    // PageRank expects store_transposed = true
    cugraph::graph_view_meta_t<vertex_t, edge_t, true, false> meta{
        h_graph.num_vertices, 
        h_graph.num_edges, 
        cugraph::graph_properties_t{h_graph.is_symmetric, false}
    };

    auto view = cugraph::graph_view_t<vertex_t, edge_t, true, false>(offsets_span, indices_span, meta);

    // Set up RMM pool to avoid fragmentation and allocation overhead
    rmm::mr::cuda_memory_resource cuda_mr;
    
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    
    free_mem = free_mem*0.95;
    free_mem = ((free_mem + 255)/256)*256;
    rmm::mr::pool_memory_resource pool_mr{cuda_mr, free_mem};
    rmm::mr::set_current_device_resource(pool_mr);

    // Precompute out-degrees (out-weights) using a custom kernel
    std::cout << "Precomputing out-degrees...\n";
    rmm::device_uvector<weight_t> d_out_degrees(h_graph.num_vertices, stream);
    cudaMemsetAsync(d_out_degrees.data(), 0, d_out_degrees.size() * sizeof(weight_t), stream);
    
    int blockSize = 256;
    int numBlocks = (h_graph.num_edges + blockSize - 1) / blockSize;
    if (numBlocks > 0) {
        compute_out_degrees_kernel<<<numBlocks, blockSize, 0, stream>>>(
            d_indices.data(), h_graph.num_edges, d_out_degrees.data());
    }
    
    auto out_degrees_span = raft::device_span<weight_t const>(d_out_degrees.data(), d_out_degrees.size());

    // PageRank parameters
    result_t alpha = 0.85;
    result_t epsilon = 1e-4;
    size_t max_iterations = 500;

    // Warm up the GPU
    std::cout << "Warming up GPU...\n";
    cugraph::pagerank<vertex_t, edge_t, weight_t, result_t, false>(
        handle, view, std::nullopt, out_degrees_span, std::nullopt, std::nullopt, 
        alpha, epsilon, max_iterations, false
    );
    handle.sync_stream();

    std::cout << "Running PageRank benchmarks...\n";
    float total_ms_pr = 0.;
    
    for (int i = 0; i < 5; i++) {
        cugraph::centrality_algorithm_metadata_t metadata;
        float ms_pr = time_gpu_ms(handle, [&]() {
            auto result = cugraph::pagerank<vertex_t, edge_t, weight_t, result_t, false>(
                handle, view, std::nullopt, out_degrees_span, std::nullopt, std::nullopt, 
                alpha, epsilon, max_iterations, false
            );
            metadata = std::get<1>(result);
        });
        std::cout << "PageRank run " << i+1 << " done in " << ms_pr << " ms (" 
                  << metadata.number_of_iterations_ << " iterations, " 
                  << ms_pr / metadata.number_of_iterations_ << " ms/iter). Converged: " 
                  << (metadata.converged_ ? "yes" : "no") << "\n";
        total_ms_pr += ms_pr;
    }

    std::cout << "Overall average PageRank time: " << total_ms_pr / 5 << " ms.\n";

    return 0;
}
