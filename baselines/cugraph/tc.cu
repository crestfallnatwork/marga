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
#include <algorithm>
#include <omp.h>
#include "csr.hpp"

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

    // Set up RMM pool to avoid fragmentation and allocation overhead
    // (Must be declared before raft::handle_t to prevent segfaults during destruction)
    rmm::mr::cuda_memory_resource cuda_mr;
    
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    
    free_mem = free_mem*0.95;
    free_mem = ((free_mem + 255)/256)*256;
    rmm::mr::pool_memory_resource pool_mr{cuda_mr, free_mem};
    auto old_mr = rmm::mr::set_current_device_resource(pool_mr);


    std::string filename = argv[1];
    
    // Import graph and force symmetrization for TC
    std::cout << "Importing and symmetrizing graph...\n";
    auto h_graph = Graph<vertex_t, edge_t>::import(filename, true, GraphLayout::CSR);

    std::cout << "Sorting CSR adjacency lists (required for TC)...\n";
    #pragma omp parallel for
    for(vertex_t i = 0; i < h_graph.num_vertices; i++) {
        edge_t start = h_graph.row_offsets[i];
        edge_t end = h_graph.row_offsets[i+1];
        std::sort(h_graph.column_indices.data() + start, h_graph.column_indices.data() + end);
    }

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

    // TC expects store_transposed = false (CSR)
    cugraph::graph_view_meta_t<vertex_t, edge_t, false, false> meta{
        h_graph.num_vertices, 
        h_graph.num_edges, 
        cugraph::graph_properties_t{h_graph.is_symmetric, false}
    };

    auto view = cugraph::graph_view_t<vertex_t, edge_t, false, false>(offsets_span, indices_span, meta);


    rmm::device_uvector<edge_t> tc_counts(h_graph.num_vertices, stream);
    auto tc_counts_span = raft::device_span<edge_t>(tc_counts.data(), tc_counts.size());

    // Warm up the GPU
    std::cout << "Warming up GPU...\n";
    cugraph::triangle_count<vertex_t, edge_t, false>(
        handle, view, std::nullopt, tc_counts_span, false);
    handle.sync_stream();

    std::cout << "Running Triangle Counting benchmarks...\n";
    float total_ms_tc = 0.;
    for (int i = 0; i < 5; i++) {
        float ms_tc = time_gpu_ms(handle, [&]() {
            cugraph::triangle_count<vertex_t, edge_t, false>(
                handle, view, std::nullopt, tc_counts_span, false);
        });
        std::cout << "TC run " << i+1 << " done in " << ms_tc << " ms.\n";
        total_ms_tc += ms_tc;
    }

    std::cout << "Overall average TC time: " << total_ms_tc / 5 << " ms.\n";

    // Read back and sum the total triangles
    std::vector<edge_t> h_counts(h_graph.num_vertices);
    raft::update_host(h_counts.data(), tc_counts.data(), h_graph.num_vertices, stream);
    handle.sync_stream();

    long long total_triangles = 0;
    for (vertex_t i = 0; i < h_graph.num_vertices; i++) {
        total_triangles += h_counts[i];
    }
    // Each triangle is counted 3 times (once for each vertex)
    std::cout << "Total triangles in graph: " << total_triangles / 3 << "\n";

    rmm::mr::set_current_device_resource(old_mr);

    return 0;
}
