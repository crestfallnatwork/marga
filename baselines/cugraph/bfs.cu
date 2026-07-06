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
#include <fstream>
#include <chrono>
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
    bool do_validation = false;
    std::string filename = "";
    std::string vertices_file = "";
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-v") {
            do_validation = true;
        } else if (arg == "-f") {
            if (i + 1 < argc) {
                vertices_file = argv[++i];
            } else {
                std::cerr << "Expected filename after -f\n";
                return 1;
            }
        } else {
            filename = arg;
        }
    }

    if (filename == "") {
        std::cerr << "Usage: " << argv[0] << " [-v] [-f <vertices.txt>] <graph.mtx>\n";
        return 1;
    }

    // Using int32_t for both handles up to 2.14B edges (directed GAP-twitter/web scale).
    // This dramatically cuts memory use.
    using vertex_t = int32_t;
    using edge_t = int32_t;
    
    // Import graph (CSR layout)
    auto h_graph = Graph<vertex_t, edge_t>::import(filename);

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

    cugraph::graph_view_meta_t<vertex_t, edge_t, false, false> meta{
        h_graph.num_vertices, 
        h_graph.num_edges, 
        cugraph::graph_properties_t{h_graph.is_symmetric, false}
    };

    auto view = cugraph::graph_view_t<vertex_t, edge_t, false, false>(offsets_span, indices_span, meta);

    // Set up RMM pool to avoid fragmentation and allocation overhead
    rmm::mr::cuda_memory_resource cuda_mr;
    
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    
    // Use half of available memory to support both 8GB and 24GB GPUs seamlessly
    free_mem = free_mem*0.95;
    free_mem = ((free_mem + 255)/256)*256;
    rmm::mr::pool_memory_resource pool_mr{cuda_mr, free_mem};
    rmm::mr::set_current_device_resource(pool_mr);

    std::vector<vertex_t> source_vertices;
    if (vertices_file != "") {
        std::ifstream vf(vertices_file);
        if (!vf.is_open()) {
            std::cerr << "Failed to open vertices file: " << vertices_file << "\n";
            return 1;
        }
        vertex_t v;
        while (vf >> v) {
            source_vertices.push_back(v - 1);
        }
    } else {
        source_vertices.push_back(0);
    }
    
    rmm::device_uvector<vertex_t> bfs_distances(h_graph.num_vertices, stream);
    rmm::device_uvector<vertex_t> d_start(1, stream);

    // Warm up the GPU
    std::cout << "Warming up GPU...\n";
    for (vertex_t src_vertex : source_vertices) {
        raft::update_device(d_start.data(), &src_vertex, 1, stream);
        cugraph::bfs<vertex_t, edge_t, false>(
            handle, view, bfs_distances.data(), nullptr,
            d_start.data(), 1, false, std::numeric_limits<vertex_t>::max(), false);
    }
    handle.sync_stream();

    std::cout << "Running BFS benchmarks...\n";
    float total_ms_bfs = 0.;
    for (vertex_t src_vertex : source_vertices) {
        raft::update_device(d_start.data(), &src_vertex, 1, stream);
        handle.sync_stream();

        float ms_bfs = time_gpu_ms(handle, [&]() {
            cugraph::bfs<vertex_t, edge_t, false>(
                handle, view, bfs_distances.data(), nullptr,
                d_start.data(), 1, false, std::numeric_limits<vertex_t>::max(), false);
        });

        std::cout << "BFS time for source " << src_vertex + 1 << ": " << ms_bfs << " ms.\n";
        total_ms_bfs += ms_bfs;

        // Simple validation check for this vertex
        if (do_validation) {
            std::vector<vertex_t> h_distances(h_graph.num_vertices);
            raft::update_host(h_distances.data(), bfs_distances.data(), h_graph.num_vertices, stream);
            handle.sync_stream();

            std::cout << "Validating BFS against CPU implementation for source " << src_vertex + 1 << "..." << std::endl;
            std::vector<vertex_t> cpu_distances(h_graph.num_vertices, std::numeric_limits<vertex_t>::max());
            std::vector<vertex_t> q;
            q.reserve(h_graph.num_vertices);
            
            cpu_distances[src_vertex] = 0;
            q.push_back(src_vertex);
            size_t head = 0;

            while (head < q.size()) {
                vertex_t u = q[head++];
                edge_t start = h_graph.row_offsets[u];
                edge_t end = h_graph.row_offsets[u+1];
                for (edge_t i = start; i < end; ++i) {
                    vertex_t v = h_graph.column_indices[i];
                    if (cpu_distances[v] == std::numeric_limits<vertex_t>::max()) {
                        cpu_distances[v] = cpu_distances[u] + 1;
                        q.push_back(v);
                    }
                }
            }

            bool valid = true;
            for (vertex_t i = 0; i < h_graph.num_vertices; ++i) {
                if (h_distances[i] != cpu_distances[i]) {
                    std::cerr << "Mismatch at vertex " << i << ": GPU " << h_distances[i] << " != CPU " << cpu_distances[i] << "\n";
                    valid = false;
                    break;
                }
            }

            if (valid) {
                std::cout << "BFS validation passed for source " << src_vertex + 1 << "! Hop counts match." << std::endl;
            } else {
                std::cout << "BFS validation failed for source " << src_vertex + 1 << "!" << std::endl;
                return 1;
            }
        }
    }
    
    if (source_vertices.size() > 1) {
        std::cout << "Overall average BFS time across all sources: " << total_ms_bfs / source_vertices.size() << " ms.\n";
    }

    return 0;
}
