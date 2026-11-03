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

#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <vector>
#include "csr.hpp"

template <typename F>
float time_gpu_ms(raft::handle_t const& handle, F&& fn) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    auto stream = handle.get_stream().get();
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
    std::string filename;
    std::string dump_path;
    std::vector<size_t> k_cores;
    int num_runs   = 5;
    bool validate  = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-v" || arg == "--validate") {
            validate = true;
        } else if (arg == "-n") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for -n\n";
                return 1;
            }
            num_runs = std::stoi(argv[++i]);
        } else if (arg == "--dump") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for --dump\n";
                return 1;
            }
            dump_path = argv[++i];
        } else if (arg == "--k-core") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for --k-core\n";
                return 1;
            }
            std::string list = argv[++i];
            size_t pos = 0;
            while (pos <= list.size()) {
                auto comma = list.find(',', pos);
                auto tok = list.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
                if (!tok.empty()) k_cores.push_back(static_cast<size_t>(std::stoull(tok)));
                if (comma == std::string::npos) break;
                pos = comma + 1;
            }
        } else if (!arg.empty() && arg[0] == '-') {
            std::cerr << "Unknown option: " << arg << "\n";
            return 1;
        } else {
            filename = arg;
        }
    }

    if (k_cores.size() > 3) {
        std::cerr << "At most 3 --k-core values are supported\n";
        return 1;
    }

    if (filename.empty() || num_runs < 1) {
        std::cerr << "Usage: " << argv[0]
                  << " <graph.mtx> [-n <runs>] [-v|--validate] [--dump <path>]"
                  << " [--k-core <k>[,<k>...]]\n";
        return 1;
    }

#ifdef KC_EDGE64
    using vertex_t = int64_t;
    using edge_t = int64_t;
#else
    using vertex_t = int32_t;
    using edge_t = int32_t;
#endif
    using weight_t = float;

    // Set up RMM pool to avoid fragmentation and allocation overhead
    // (Must be declared before raft::handle_t to prevent segfaults during destruction)
    rmm::mr::cuda_memory_resource cuda_mr;

    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);

    free_mem = free_mem * 0.95;
    free_mem = ((free_mem + 255) / 256) * 256;
    rmm::mr::pool_memory_resource pool_mr{cuda_mr, free_mem};
    auto old_mr = rmm::mr::set_current_device_resource(pool_mr);

    // core_number requires an undirected (symmetric) graph
    std::cout << "Importing and symmetrizing graph...\n";
    auto import_t0 = std::chrono::steady_clock::now();
    auto h_graph = Graph<vertex_t, edge_t>::import(filename, true, GraphLayout::CSR);
    auto import_t1 = std::chrono::steady_clock::now();
    std::cout << "Import (parse+symmetrize): "
              << std::chrono::duration<double, std::milli>(import_t1 - import_t0).count()
              << " ms\n";
    std::cout << "Loaded: " << h_graph.num_vertices << " vertices, " << h_graph.num_edges
              << " edges (" << (h_graph.is_symmetric ? "symmetric" : "directed") << ")\n";

    raft::handle_t handle;
    auto stream = handle.get_stream().get();

    std::cout << "Transferring graph to GPU...\n";
    rmm::device_uvector<edge_t> d_offsets(h_graph.row_offsets.size(), stream);
    rmm::device_uvector<vertex_t> d_indices(h_graph.column_indices.size(), stream);

    raft::update_device(d_offsets.data(), h_graph.row_offsets.data(), h_graph.row_offsets.size(), stream);
    raft::update_device(d_indices.data(), h_graph.column_indices.data(), h_graph.column_indices.size(), stream);
    handle.sync_stream();
    std::cout << "Transfer: "
              << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - import_t1).count()
              << " ms\n";

    auto offsets_span = raft::device_span<edge_t const>(d_offsets.data(), d_offsets.size());
    auto indices_span = raft::device_span<vertex_t const>(d_indices.data(), d_indices.size());

    cugraph::graph_view_meta_t<vertex_t, edge_t, false, false> meta{
        h_graph.num_vertices,
        h_graph.num_edges,
        cugraph::graph_properties_t{h_graph.is_symmetric, false}
    };

    auto view = cugraph::graph_view_t<vertex_t, edge_t, false, false>(offsets_span, indices_span, meta);

    rmm::device_uvector<edge_t> core_numbers(h_graph.num_vertices, stream);

    // Warm up the GPU
    std::cout << "Warming up GPU...\n";
    cugraph::core_number<vertex_t, edge_t, false>(
        handle, view, core_numbers.data(), cugraph::k_core_degree_type_t::IN,
        0, std::numeric_limits<size_t>::max(), false);
    handle.sync_stream();

    std::cout << "Running K-Core benchmarks...\n";
    float total_ms_kc = 0.;
    for (int i = 0; i < num_runs; i++) {
        float ms_kc = time_gpu_ms(handle, [&]() {
            cugraph::core_number<vertex_t, edge_t, false>(
                handle, view, core_numbers.data(), cugraph::k_core_degree_type_t::IN,
                0, std::numeric_limits<size_t>::max(), false);
        });
        std::cout << "KC run " << i + 1 << " done in " << ms_kc << " ms.\n";
        total_ms_kc += ms_kc;
    }

    std::cout << "Overall average KC time: " << total_ms_kc / num_runs << " ms.\n";

    // Read back and summarize the core numbers
    std::vector<edge_t> h_core_numbers(h_graph.num_vertices);
    raft::update_host(h_core_numbers.data(), core_numbers.data(), h_graph.num_vertices, stream);
    handle.sync_stream();

    long long core_sum = 0;
    edge_t max_core = 0;
    for (vertex_t i = 0; i < h_graph.num_vertices; i++) {
        core_sum += h_core_numbers[i];
        max_core = std::max(max_core, h_core_numbers[i]);
    }

    // Core values need not be contiguous; count distinct values
    rmm::device_uvector<edge_t> sorted_core_numbers(h_graph.num_vertices, stream);
    thrust::copy(handle.get_thrust_policy(), core_numbers.begin(), core_numbers.end(),
                 sorted_core_numbers.begin());
    thrust::sort(handle.get_thrust_policy(), sorted_core_numbers.begin(), sorted_core_numbers.end());
    auto num_distinct = thrust::unique_count(
        handle.get_thrust_policy(), sorted_core_numbers.begin(), sorted_core_numbers.end());

    std::cout << "Sum of core numbers: " << core_sum << "\n";
    std::cout << "Max core number: " << max_core << "\n";
    std::cout << "Number of distinct core values: " << num_distinct << "\n";

    if (validate) {
        auto n_print = std::min<vertex_t>(64, h_graph.num_vertices);
        std::cout << "Core numbers (first " << n_print << " vertices):";
        for (vertex_t i = 0; i < n_print; i++) {
            std::cout << " " << h_core_numbers[i];
        }
        std::cout << "\n";
    }

    if (!dump_path.empty()) {
        std::ofstream out(dump_path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(h_core_numbers.data()),
                  h_core_numbers.size() * sizeof(edge_t));
        out.close();
        if (!out) {
            std::cerr << "Failed to write core numbers to " << dump_path << "\n";
            return 1;
        }
        std::cout << "Dump: " << dump_path << " (" << h_graph.num_vertices
                  << (sizeof(edge_t) == 8 ? " int64 values)\n" : " int32 values)\n");
    }

    for (size_t k : k_cores) {
        auto [srcs, dsts, weights] = cugraph::k_core<vertex_t, edge_t, weight_t, false>(
            handle, view, std::nullopt, k, cugraph::k_core_degree_type_t::IN,
            raft::device_span<edge_t const>(core_numbers.data(), core_numbers.size()), false);
        handle.sync_stream();

        rmm::device_uvector<vertex_t> sorted_srcs(srcs.size(), stream);
        thrust::copy(handle.get_thrust_policy(), srcs.begin(), srcs.end(), sorted_srcs.begin());
        thrust::sort(handle.get_thrust_policy(), sorted_srcs.begin(), sorted_srcs.end());
        auto distinct_srcs = thrust::unique_count(
            handle.get_thrust_policy(), sorted_srcs.begin(), sorted_srcs.end());

        long long members = 0, arcs = 0;
        for (vertex_t v = 0; v < h_graph.num_vertices; v++) {
            if (static_cast<size_t>(h_core_numbers[v]) < k) continue;
            members++;
            for (edge_t e = h_graph.row_offsets[v]; e < h_graph.row_offsets[v + 1]; e++) {
                vertex_t u = h_graph.column_indices[e];
                if (static_cast<size_t>(h_core_numbers[u]) >= k) arcs++;
            }
        }

        std::cout << "K-core " << k << ": " << srcs.size() << " edges, " << distinct_srcs
                  << " distinct source vertices\n";
        std::cout << "Expected: members=" << members << ", arcs=" << arcs << " ("
                  << arcs / 2 << " undirected)\n";
        bool ok = distinct_srcs == static_cast<size_t>(members) &&
                  srcs.size() == static_cast<size_t>(arcs);
        std::cout << "K-core check: " << (ok ? "PASS" : "FAIL") << "\n";
    }

    rmm::mr::set_current_device_resource(old_mr);

    return 0;
}
