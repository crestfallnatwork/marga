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
    uint iterations = 5;

    auto cli = group(value("graph file", graph_file) % "MTX graph file to process",
                     option("-v", "--validate").set(validate) %
                         "validate GPU results against parallel CPU TC",
                     option("-i", "--iterations").set(iterations) % "number of iterations to run");

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

        auto tc_time = 0.s;
        uint64_t res = 0;
        for (uint i = 0; i < iterations; i++) {
            pictor::graphs::TC tc(ctx);
            auto t1      = high_resolution_clock::now();
            res = tc.run(g);
            auto t2      = high_resolution_clock::now();
            tc_time += t2 - t1;
            ctx.wait_idle();
        }
        std::cout << std::format("TC Took: {}\n", tc_time / iterations);
        std::cout << std::format("TC: {}\n", res);
        if (validate) {
            std::cout << "validation not implemented yet\n";
        }
    } else {
        std::cout << make_man_page(cli, argv[0]);
        return 1;
    }

    return 0;
}
