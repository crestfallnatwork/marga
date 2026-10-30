// SPDX-License-Identifier: GPL-2.0-only
// Author: Himanshu Gupta <himanshu3@iisc.ac.in>

#include "cc.hpp"
#include "cc_validation.hpp"
#include "clipp.h"
#include "csr.hpp"
#include "pictor_core/context/context.hpp"
#include "pictor_core/memory/heap_allocator.hpp"

#include <chrono>
#include <format>
#include <iostream>
#include <string>

int main(int argc, char* argv[])
{
    using namespace clipp;
    using namespace std::chrono;

    std::string graph_file;
    bool        validate = false;
    bool        sym      = false;
    uint32_t    num_runs = 1;

    auto cli = group(
        value("graph file", graph_file) % "MTX graph file to process",
        (option("-n", "--num-runs") & value("number of runs", num_runs)) %
            "number of times to run the benchmark",
        option("-v", "--validate").set(validate) %
            "validate GPU components against a CPU union-find",
        option("-s", "--sym").set(sym) %
            "symmetric view: import symmetrized and process each undirected edge once");

    if (!parse(argc, argv, cli)) {
        std::cout << make_man_page(cli, argv[0]);
        return 1;
    }

    auto ctx_res = pictor::Context::create();
    if (!ctx_res.ok()) {
        std::cerr << "Failed to create context" << std::endl;
        return 1;
    }
    auto& ctx = *ctx_res.val();

    std::cout << "Importing graph as CSR: " << graph_file << "..." << std::endl;
    auto g = pictor::graphs::Graph::import(ctx, graph_file, sym, pictor::graphs::GraphLayout::CSR);

    size_t free_mem = ctx.get_device_free_memory() * 0.95;
    auto   dev_allocator =
        std::move(pictor::HeapAllocator::create(&ctx, ctx.root_pool_allocator.get(), free_mem).val());
    auto host_allocator =
        std::move(pictor::HeapAllocator::create(&ctx, ctx.root_pinned_allocator.get(), free_mem).val());
    ctx.set_device_allocator(std::move(dev_allocator));
    ctx.set_host_allocator(std::move(host_allocator));

    std::cout << std::format("Graph: {} vertices, {} edges ({} view)\n", g.num_vertices, g.num_edges,
                             g.symmetrized ? "symmetric" : "directed");
    if (g.symmetrized && !sym)
        std::cout << "Note: the file is symmetric, so this run already sees both directions; "
                     "use -s to cull one direction.\n";

    pictor::graphs::CC cc(ctx);

    double total_ms = 0.0;
    double loop_ms  = 0.0;
    double init_ms  = 0.0;
    uint32_t rounds = 0;

    std::cout << "Running CC " << num_runs << " times...\n";
    for (uint32_t run = 0; run < num_runs; ++run) {
        const auto t1 = high_resolution_clock::now();
        rounds        = cc.run(g, sym);
        const auto t2 = high_resolution_clock::now();

        const double ms = duration<double, std::milli>(t2 - t1).count();
        total_ms += ms;
        loop_ms += cc.loop_ms();
        init_ms += cc.init_ms();
        std::cout << std::format(
            "Run {} - CC Total: {:.3f} ms (init {:.3f}, loop {:.3f}, {} rounds)\n", run + 1, ms,
            cc.init_ms(), cc.loop_ms(), rounds);
        ctx.wait_idle();
    }

    std::cout << std::format("\n=== Average Results ({} runs) ===\n", num_runs);
    std::cout << std::format("  Average CC loop:  {:.3f} ms\n", loop_ms / num_runs);
    std::cout << std::format("  Average CC init:  {:.3f} ms\n", init_ms / num_runs);
    std::cout << std::format("  Average CC total: {:.3f} ms\n", total_ms / num_runs);
    std::cout << std::format("  Rounds:           {}\n", rounds);

    const uint64_t components = cc.component_count();
    std::cout << std::format("Components: {}\n", components);

    if (validate) {
        std::cout << "Running CPU Validation..." << std::endl;
        const auto val_start = high_resolution_clock::now();
        const auto cpu       = pictor::graphs::CCValidator::compute(g.row_offsets, g.column_indices);
        const auto val_end   = high_resolution_clock::now();
        const double val_ms  = duration<double, std::milli>(val_end - val_start).count();

        const auto roots = cc.resolve_roots();
        const bool ok    = pictor::graphs::CCValidator::verify(roots, cpu);

        std::cout << std::format("CPU components: {} ({} ms)\n", cpu.components, val_ms);
        std::cout << (ok ? "SUCCESS: GPU and CPU component partitions match.\n"
                         : "FAILURE: GPU and CPU component partitions differ.\n");
        if (!ok)
            return 2;
    }

    return 0;
}
