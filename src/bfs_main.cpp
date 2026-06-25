// SPDX-License-Identifier: GPL-2.0-only
// Author: Himanshu Gupta <himanshu3@iisc.ac.in>

#include "clipp.h"
#include "bfs.hpp"
#include "csr.hpp"
#include "bfs_validation.hpp"
#include "pictor_core/context/context.hpp"

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cstdint>
#include <chrono>
#include <numeric>
#include <iomanip>

int main(int argc, char* argv[]) {
    using namespace clipp;
    
    std::string graph_file;
    uint32_t src_vertex = 0;
    std::string vertices_file;
    bool validate = false;

    auto cli = (
        value("graph file", graph_file),
        (
            (option("-s", "--src") & value("source vertex", src_vertex)) % "source vertex to start BFS from" |
            (option("-f", "--file") & value("vertices file", vertices_file)) % "file containing source vertices (one per line)"
        ),
        option("-v", "--validate").set(validate) % "validate GPU results against parallel CPU BFS"
    );

    if (parse(argc, argv, cli)) {
        auto ctx_res = pictor::Context::create();
        if (!ctx_res.ok()) {
            std::cerr << "Failed to create context" << std::endl;
            return 1;
        }
        auto& ctx = *ctx_res.val();

        std::cout << "Importing graph: " << graph_file << "..." << std::endl;
        auto import_start = std::chrono::high_resolution_clock::now();
        auto g = pictor::graphs::Graph::import(ctx, graph_file);
        auto import_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> import_elapsed = import_end - import_start;
        std::cout << "Import took: " << import_elapsed.count() << "s" << std::endl;

        std::vector<uint32_t> sources;
        if (!vertices_file.empty()) {
            std::ifstream infile(vertices_file);
            uint32_t v;
            while (infile >> v) sources.push_back(v);
        } else {
            sources.push_back(src_vertex);
        }

        pictor::graphs::BFS bfs(ctx);
        std::vector<pictor::graphs::BFSTimings> all_gpu_timings;
        std::vector<pictor::graphs::ValidationTimings> all_cpu_timings;

        std::cout << "Running BFS for " << sources.size() << " source(s)..." << std::endl;

        for (uint32_t src : sources) {
            std::cout << "\n--- Source: " << src << " ---" << std::endl;
            
            ctx.wait_idle(); // Ensure GPU is clean before next run

            auto [gpu_distances, gpu_t] = bfs.run(g, src);
            all_gpu_timings.push_back(gpu_t);
            std::cout << "  GPU Run Total: " << gpu_t.total_ms << " ms" << std::endl;

            if (validate) {
                std::cout << "Running CPU Validation..." << std::endl;
                auto [cpu_distances, cpu_t] = pictor::graphs::BFSValidator::run(g, src);
                all_cpu_timings.push_back(cpu_t);
                std::cout << "CPU Total: " << cpu_t.traversal_ms << " ms" << std::endl;
                
                pictor::graphs::BFSValidator::verify(g, src, gpu_distances, cpu_distances);
            }
        }

        // Aggregate and print averages
        auto print_avg_gpu = [&](const std::vector<pictor::graphs::BFSTimings>& timings) {
            pictor::graphs::BFSTimings avg;
            for (const auto& t : timings) {
                avg.alloc_ms += t.alloc_ms;
                avg.init_ms += t.init_ms;
                avg.traversal_ms += t.traversal_ms;
                avg.inner_sync_ms += t.inner_sync_ms;
                avg.pci_readback_ms += t.pci_readback_ms;
                avg.total_ms += t.total_ms;
            }
            double n = static_cast<double>(timings.size());
            avg.alloc_ms /= n; avg.init_ms /= n; avg.traversal_ms /= n;
            avg.inner_sync_ms /= n; avg.pci_readback_ms /= n; avg.total_ms /= n;

            std::cout << "\n=== GPU BFS Average Results (" << timings.size() << " runs) ===" << std::endl;
            std::cout << "  VMA Allocations:     " << avg.alloc_ms << " ms" << std::endl;
            std::cout << "  Initialization (H2D): " << avg.init_ms << " ms" << std::endl;
            std::cout << "  Traversal (GPU):     " << avg.traversal_ms << " ms (Inner sync: " << avg.inner_sync_ms << " ms)" << std::endl;
            std::cout << "  Final PCIe Readback: " << avg.pci_readback_ms << " ms" << std::endl;
            std::cout << "  --------------------------------" << std::endl;
            std::cout << "  AVERAGE TOTAL:       " << avg.total_ms << " ms" << std::endl;
        };

        auto print_avg_cpu = [&](const std::vector<pictor::graphs::ValidationTimings>& timings) {
            double avg_traversal = 0;
            for (const auto& t : timings) avg_traversal += t.traversal_ms;
            avg_traversal /= timings.size();

            std::cout << "\n=== CPU BFS Average Results (" << timings.size() << " runs) ===" << std::endl;
            std::cout << "  Traversal (CPU):     " << avg_traversal << " ms" << std::endl;
        };

        std::cout << std::fixed << std::setprecision(3);
        print_avg_gpu(all_gpu_timings);
        if (!all_cpu_timings.empty()) {
            print_avg_cpu(all_cpu_timings);
        }
        std::cout << "========================================================\n" << std::endl;

    } else {
        std::cout << make_man_page(cli, argv[0]);
        return 1;
    }

    return 0;
}
