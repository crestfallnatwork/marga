// SPDX-License-Identifier: GPL-2.0-only
// Author: Himanshu Gupta <himanshu3@iisc.ac.in>

#include "clipp.h"
#include "csr.hpp"
#include "pictor_core/context/context.hpp"
#include "pictor_core/memory/heap_allocator.hpp"
#include "tc.hpp"

#include <chrono>
#include <iostream>
#include <string>

int main(int argc, char* argv[])
{
    using namespace clipp;
    using namespace std::chrono;
    using namespace std::chrono_literals;

    std::string graph_file;
    bool validate   = false;
    uint32_t num_runs = 1;

    auto cli = group(value("graph file", graph_file) % "MTX graph file to process",
                     (option("-n", "--num-runs") & value("number of runs", num_runs)) %
                         "number of times to run the benchmark",
                     option("-v", "--validate").set(validate) %
                         "validate GPU results against parallel CPU TC");

    if (parse(argc, argv, cli)) {
        auto ctx_res = pictor::Context::create();
        if (!ctx_res.ok()) {
            std::cerr << "Failed to create context" << std::endl;
            return 1;
        }
        auto& ctx = *ctx_res.val();

        uint64_t traingles_gpu = 0;

        std::cout << "Importing graph as CSR: " << graph_file << "..." << std::endl;
        auto g =
            pictor::graphs::Graph::import(ctx, graph_file, true, pictor::graphs::GraphLayout::CSR);

        size_t free_mem    = ctx.get_device_free_memory() * 0.95;
        auto dev_allocator = std::move(
            pictor::HeapAllocator::create(&ctx, ctx.root_pool_allocator.get(), free_mem).val());
        auto host_allocator = std::move(
            pictor::HeapAllocator::create(&ctx, ctx.root_pinned_allocator.get(), free_mem).val());
        ctx.set_device_allocator(std::move(dev_allocator));
        ctx.set_host_allocator(std::move(host_allocator));

        double total_time_ms = 0;
        uint64_t res = 0;
        
        std::cout << "Running TC " << num_runs << " times...\n";
        for (uint32_t run = 0; run < num_runs; run++) {
            pictor::graphs::TC tc(ctx);
            
            auto t1 = high_resolution_clock::now();
            res = tc.run(g);
            auto t2 = high_resolution_clock::now();
            
            double time_ms = duration<double, std::milli>(t2 - t1).count();
            std::cout << "Run " << run + 1 << " - TC Run Total: " << time_ms << " ms, Output: " << res << "\n";
            
            total_time_ms += time_ms;
            ctx.wait_idle();
        }
        
        std::cout << "\n=== Average Results (" << num_runs << " runs) ===\n";
        std::cout << "  Average TC Run Total: " << total_time_ms / num_runs << " ms\n";
        std::cout << "======================================\n\n";
        if (validate) {
            std::cout << "validation not implemented yet\n";
        }
    } else {
        std::cout << make_man_page(cli, argv[0]);
        return 1;
    }

    return 0;
}
