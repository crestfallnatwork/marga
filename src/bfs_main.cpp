// SPDX-License-Identifier: GPL-2.0-only
// Author: Himanshu Gupta <himanshu3@iisc.ac.in>

#include "bfs.hpp"
#include "bfs_validation.hpp"
#include "clipp.h"
#include "csr.hpp"
#include "pictor_core/context/context.hpp"
#include "pictor_core/memory/heap_allocator.hpp"

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

int main(int argc, char* argv[])
{
    using namespace clipp;

    std::string graph_file;
    uint32_t src_vertex = 0;
    std::string vertices_file;
    bool validate = false;

    auto cli = (value("graph file", graph_file),
                ((option("-s", "--src") & value("source vertex", src_vertex)) %
                     "source vertex to start BFS from" |
                 (option("-f", "--file") & value("vertices file", vertices_file)) %
                     "file containing source vertices (one per line)"),
                option("-v", "--validate").set(validate) %
                    "validate GPU results against parallel CPU BFS");

    if (parse(argc, argv, cli)) {
        auto ctx_res = pictor::Context::create();
        if (!ctx_res.ok()) {
            std::cerr << "Failed to create context" << std::endl;
            return 1;
        }
        auto& ctx = *ctx_res.val();

        std::cout << "Importing graph: " << graph_file << "..." << std::endl;
        auto import_start = std::chrono::high_resolution_clock::now();
        auto g            = pictor::graphs::Graph::import(ctx, graph_file);
        auto import_end   = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> import_elapsed = import_end - import_start;
        std::cout << "Import took: " << import_elapsed.count() << "s" << std::endl;

        std::vector<uint32_t> sources;
        if (!vertices_file.empty()) {
            std::ifstream infile(vertices_file);
            uint32_t v;
            while (infile >> v)
                sources.push_back(v);
        } else {
            sources.push_back(src_vertex);
        }

        size_t free_mem = ctx.get_device_free_memory() * 0.95;
        ctx.set_device_allocator(std::move(
            pictor::HeapAllocator::create(&ctx, ctx.root_pool_allocator.get(), free_mem).val()));
        ctx.set_host_allocator(std::move(
            pictor::HeapAllocator::create(&ctx, ctx.root_pinned_allocator.get(), free_mem).val()));

        pictor::graphs::BFS bfs(ctx);
        bfs.allocate(g.num_vertices, g.num_edges);
        
        std::vector<pictor::graphs::ValidationTimings> all_cpu_timings;
        double total_time_ms = 0;

        std::cout << "Running BFS for " << sources.size() << " source(s)..." << std::endl;

        for (uint32_t src : sources) {
            std::cout << "\n--- Source: " << src << " ---" << std::endl;

            ctx.wait_idle(); // Ensure GPU is clean before next run

            auto run_start = std::chrono::high_resolution_clock::now();
            bfs.run(g, src);
            auto run_end = std::chrono::high_resolution_clock::now();

            double time_ms = std::chrono::duration<double, std::milli>(run_end - run_start).count();
            std::cout << "  GPU Run Total: " << time_ms << " ms" << std::endl;
            total_time_ms += time_ms;

            if (validate) {
                auto gpu_distances = bfs.copy_to_host(g.num_vertices);
                std::span<uint32_t> gpu_distances_span(gpu_distances.data(), g.num_vertices);
                
                std::cout << "Running CPU Validation..." << std::endl;
                auto [cpu_distances, cpu_t] = pictor::graphs::BFSValidator::run(g, src);
                all_cpu_timings.push_back(cpu_t);
                std::cout << "CPU Total: " << cpu_t.traversal_ms << " ms" << std::endl;

                pictor::graphs::BFSValidator::verify(g, src, gpu_distances_span, cpu_distances);
            }
        }

        std::cout << "\n=== GPU BFS Average Results (" << sources.size() << " runs) ===" << std::endl;
        std::cout << "  AVERAGE TOTAL:       " << total_time_ms / sources.size() << " ms" << std::endl;

        auto print_avg_cpu = [&](const std::vector<pictor::graphs::ValidationTimings>& timings) {
            double avg_traversal = 0;
            for (const auto& t : timings)
                avg_traversal += t.traversal_ms;
            avg_traversal /= timings.size();

            std::cout << "\n=== CPU BFS Average Results (" << timings.size()
                      << " runs) ===" << std::endl;
            std::cout << "  Traversal (CPU):     " << avg_traversal << " ms" << std::endl;
        };

        std::cout << std::fixed << std::setprecision(3);
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
