// SPDX-License-Identifier: GPL-2.0-only
// Author: Himanshu Gupta <himanshu3@iisc.ac.in>

#include "clipp.h"
#include "csr.hpp"
#include "kc.hpp"
#include "kc_validation.hpp"
#include "pictor_core/context/context.hpp"
#include "pictor_core/memory/heap_allocator.hpp"

#include <algorithm>
#include <chrono>
#include <format>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

int main(int argc, char* argv[])
{
    using namespace clipp;
    using namespace std::chrono;

    std::string graph_file;
    bool        validate         = false;
    bool        keep_self_loops  = false;
    uint32_t    num_runs         = 1;
    uint32_t    phases_per_submit = 32;
    uint32_t    k_core           = 0;
    std::string dump_core;

    auto cli = group(
        value("graph file", graph_file) % "MTX graph file to process",
        (option("-n", "--num-runs") & value("number of runs", num_runs)) %
            "number of times to run the benchmark",
        (option("-b", "--batch") & value("phases per submit", phases_per_submit)) %
            "device phases per host submission (1 = one fence wait per phase)",
        (option("--k-core") & value("k", k_core)) %
            "also check the k-core subgraph for this k (0: pick two from the result)",
        (option("--dump-core") & value("path", dump_core)) %
            "write the core numbers as a raw uint32 array (for cross-system checks)",
        option("-v", "--validate").set(validate) %
            "validate GPU core numbers against the exact CPU decomposition",
        option("--keep-self-loops").set(keep_self_loops) %
            "keep u == v entries (default: dropped, matching the undirected "
            "simple-graph definition)");

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
    const auto import_t0 = high_resolution_clock::now();
    auto g = pictor::graphs::Graph::import(ctx, graph_file, true, pictor::graphs::GraphLayout::CSR,
                                           !keep_self_loops);
    std::cout << std::format("Import: {:.3f} ms\n",
                             duration<double, std::milli>(high_resolution_clock::now() - import_t0)
                                 .count());

    size_t free_mem = ctx.get_device_free_memory() * 0.95;
    auto   dev_allocator =
        std::move(pictor::HeapAllocator::create(&ctx, ctx.root_pool_allocator.get(), free_mem).val());
    auto host_allocator =
        std::move(pictor::HeapAllocator::create(&ctx, ctx.root_pinned_allocator.get(), free_mem).val());
    ctx.set_device_allocator(std::move(dev_allocator));
    ctx.set_host_allocator(std::move(host_allocator));

    std::cout << std::format("Graph: {} vertices, {} edges (symmetric view, self-loops {})\n",
                             g.num_vertices, g.num_edges,
                             keep_self_loops ? "kept" : "dropped");
    std::cout << std::format("Device pool: {:.2f} GiB\n", free_mem / 1073741824.0);

    pictor::graphs::KC kc(ctx);

    double   total_ms = 0.0;
    double   loop_ms  = 0.0;
    double   init_ms  = 0.0;
    uint64_t phases   = 0;
    uint32_t seeds    = 0;
    uint32_t max_core = 0;
    uint64_t peak     = 0;
    std::vector<uint32_t> core;

    std::cout << std::format("Running k-core decomposition {} times ({} phases per submit)...\n",
                             num_runs, phases_per_submit);
    for (uint32_t run = 0; run < num_runs; ++run) {
        const auto t1 = high_resolution_clock::now();
        phases        = kc.run(g, phases_per_submit);
        const auto t2 = high_resolution_clock::now();

        const double ms = duration<double, std::milli>(t2 - t1).count();
        core            = kc.copy_core();

        uint64_t checksum = std::accumulate(core.begin(), core.end(), uint64_t{0});
        total_ms += ms;
        loop_ms += kc.loop_ms();
        init_ms += kc.init_ms();
        seeds    = kc.thresholds();
        max_core = kc.max_core();
        peak     = kc.frontier_peak();

        std::cout << std::format(
            "Run {} - KC Total: {:.3f} ms (init {:.3f}, loop {:.3f}, {} phases, {} threshold "
            "seeds, max core {}, checksum {})\n",
            run + 1, ms, kc.init_ms(), kc.loop_ms(), phases, seeds, max_core, checksum);
        ctx.wait_idle();
    }

    std::cout << std::format("\n=== Average Results ({} runs) ===\n", num_runs);
    std::cout << std::format("  Average KC total:  {:.3f} ms\n", total_ms / num_runs);
    std::cout << std::format("  Average KC init:   {:.3f} ms\n", init_ms / num_runs);
    std::cout << std::format("  Average KC loop:   {:.3f} ms\n", loop_ms / num_runs);
    std::cout << std::format("  Phases:            {}\n", phases);
    std::cout << std::format("  Threshold seeds:   {}\n", seeds);
    std::cout << std::format("  Max core:          {}\n", max_core);
    std::cout << std::format("  Frontier entries:  {} of {} ({:.2f} GiB)\n", peak,
                             kc.vertex_count(), static_cast<double>(peak) * 4.0 / 1073741824.0);

    if (!dump_core.empty() && !core.empty()) {
        std::ofstream out(dump_core, std::ios::binary);
        out.write(reinterpret_cast<const char*>(core.data()),
                  static_cast<std::streamsize>(core.size() * sizeof(uint32_t)));
        out.close();
        std::cout << std::format("Dump: {} ({} uint32 values)\n", dump_core, core.size());
    }

    if (validate) {
        std::cout << "Running CPU Validation..." << std::endl;
        const auto val_t0 = high_resolution_clock::now();
        const auto cpu =
            pictor::graphs::KCValidator::compute(g.row_offsets, g.column_indices);
        const double val_ms =
            duration<double, std::milli>(high_resolution_clock::now() - val_t0).count();

        const auto cmp = pictor::graphs::KCValidator::compare(core, cpu);
        std::cout << std::format(
            "CPU reference: {} vertices, max core {}, {} isolated; {:.3f} ms\n", cmp.checked,
            cpu.max_core, cpu.isolated, val_ms);
        if (cmp.mismatches != 0) {
            std::cout << std::format("  first mismatch at vertex {}: gpu {} vs cpu {}\n",
                                     cmp.first_vertex, cmp.first_gpu, cmp.first_cpu);
        }
        std::cout << std::format("  core numbers compared exactly: {} mismatches\n",
                                 cmp.mismatches);

        const uint32_t k_mid = std::max<uint32_t>(1, cmp.max_core / 2);
        for (uint32_t k : {k_mid, cmp.max_core, k_core}) {
            if (k == 0)
                continue;
            uint64_t members = 0, min_degree = 0;
            const bool ok = pictor::graphs::KCValidator::verify_k_core(
                g.row_offsets, g.column_indices, core, k, members, min_degree);
            std::cout << std::format(
                "  {}-core: {} vertices, min induced degree {}{}\n", k, members, min_degree,
                ok ? "" : " (NOT a k-core)");
        }

        std::cout << (cmp.mismatches == 0
                          ? "SUCCESS: GPU and CPU core numbers match exactly.\n"
                          : "FAILURE: GPU and CPU core numbers differ.\n");
        if (cmp.mismatches != 0)
            return 2;
    }

    return 0;
}
