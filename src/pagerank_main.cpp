// SPDX-License-Identifier: GPL-2.0-only
// Author: Himanshu Gupta <himanshu3@iisc.ac.in>

#include "clipp.h"
#include "csr.hpp"
#include "pagerank.hpp"
#include "pictor_core/context/context.hpp"
#include "pictor_core/memory/heap_allocator.hpp"

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>

int main(int argc, char* argv[])
{
    using namespace clipp;

    std::string graph_file;
    uint32_t num_runs = 1;
    bool validate = false;

    auto cli = group(value("graph file", graph_file) % "MTX graph file to process",
                     (option("-n", "--num-runs") & value("number of runs", num_runs)) %
                         "number of times to run the benchmark",
                     option("-v", "--validate").set(validate) %
                         "validate GPU results against parallel CPU PageRank");

    if (parse(argc, argv, cli)) {
        auto ctx_res = pictor::Context::create();
        if (!ctx_res.ok()) {
            std::cerr << "Failed to create context" << std::endl;
            return 1;
        }
        auto& ctx = *ctx_res.val();

        std::vector<float> pr_gpu;

        {
            std::cout << "Importing graph as CSC: " << graph_file << "..." << std::endl;
            auto g = pictor::graphs::Graph::import(ctx, graph_file, false,
                                                   pictor::graphs::GraphLayout::CSC);
            
            size_t free_mem = ctx.get_device_free_memory() * 0.95;
            ctx.set_device_allocator(std::move(
                pictor::HeapAllocator::create(&ctx, ctx.root_pool_allocator.get(), free_mem).val()));
            ctx.set_host_allocator(std::move(
                pictor::HeapAllocator::create(&ctx, ctx.root_pinned_allocator.get(), free_mem).val()));

            pictor::graphs::PageRank pr(ctx);
            pr.compute_out_degree(g, pictor::graphs::GraphLayout::CSC);
            
            double total_time_ms = 0;
            double total_time_per_iter = 0;
            uint32_t total_pr_iters = 0;
            
            std::cout << "Running PageRank " << num_runs << " times...\n";

            for (uint32_t run = 0; run < num_runs; run++) {
                ctx.wait_idle(); // Ensure GPU is clean before next run

                auto run_start = std::chrono::high_resolution_clock::now();
                uint32_t iterations = pr.run(g, 1e-4);
                auto run_end = std::chrono::high_resolution_clock::now();
                
                double time_ms = std::chrono::duration<double, std::milli>(run_end - run_start).count();
                double time_per_iter = time_ms / iterations;
                
                std::cout << "Run " << run + 1 << " - PR Run Total: " << time_ms 
                          << " ms, Iterations: " << iterations 
                          << ", Time/Iter: " << time_per_iter << " ms\n";
                          
                total_time_ms += time_ms;
                total_time_per_iter += time_per_iter;
                total_pr_iters += iterations;
                
                if (run == num_runs - 1) {
                    pr_gpu = pr.copy_to_host(g.num_vertices);
                }
            }
            
            std::cout << "\n=== Average Results (" << num_runs << " runs) ===\n";
            std::cout << "  Average PR Run Total: " << total_time_ms / num_runs << " ms\n";
            std::cout << "  Average Time/Iter:    " << total_time_per_iter / num_runs << " ms\n";
            std::cout << "  Average Iterations:   " << static_cast<double>(total_pr_iters) / num_runs << "\n";
            std::cout << "======================================\n\n";
        }

        if (validate) {
            std::cout << "Running sequential push iteration to validate fixed point "
                         "behaviour:"
                      << std::endl;
            auto g = pictor::graphs::Graph::import(ctx, graph_file, false,
                                                   pictor::graphs::GraphLayout::CSR);
            std::vector<double> pr_cpu(g.num_vertices, 0.0);
            double dangling  = 0.;
            double all_error = 0.;
            for (uint32_t vertex = 0; vertex < g.num_vertices; vertex++) {
                uint64_t edges = g.row_offsets[vertex + 1] - g.row_offsets[vertex];
                for (uint64_t dst : g.column_indices.subspan(g.row_offsets[vertex], edges)) {
                    pr_cpu[dst] += (static_cast<double>(pr_gpu[vertex]) / edges);
                }
            }
            dangling = 0;
            for (uint32_t vertex = 0; vertex < g.num_vertices; vertex++) {
                uint64_t edges = g.row_offsets[vertex + 1] - g.row_offsets[vertex];
                if (edges == 0) {
                    dangling += static_cast<double>(pr_gpu[vertex]);
                }
            }
            std::cout << "CPU Dangling sum (double): " << dangling << std::endl;
            for (uint32_t vertex = 0; vertex < g.num_vertices; vertex++) {
                auto csr_v = ((1. - 0.85) / float(g.num_vertices)) +
                             0.85 * (pr_cpu[vertex] + (dangling / g.num_vertices));
                auto err   = std::abs(csr_v - pr_gpu[vertex]);
                all_error += double(err);
                if (err > 1e-4) {
                    std::cout << std::format(
                                     "Didn't converge, vertex: {}, csc val: {}, new val: {}",
                                     vertex, pr_gpu[vertex], csr_v)
                              << std::endl;
                }
            }
            if (all_error > 1e-4) {
                std::cout << std::format("Didn't converge, error: {}\n", all_error);
            }
        }

    } else {
        std::cout << make_man_page(cli, argv[0]);
        return 1;
    }

    return 0;
}
