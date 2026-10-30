// SPDX-License-Identifier: GPL-2.0-only
// Author: Himanshu Gupta <himanshu3@iisc.ac.in>

#include "bc.hpp"
#include "bc_validation.hpp"
#include "clipp.h"
#include "csr.hpp"
#include "pictor_core/context/context.hpp"
#include "pictor_core/memory/heap_allocator.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <format>
#include <iostream>
#include <string>
#include <vector>

namespace {
// Vertices read back per source for the per-run checksum and for validation.
constexpr uint64_t cChecksumWindow  = 8192;
constexpr uint64_t cValidationWindow = 1ull << 20;
} // namespace

int main(int argc, char* argv[])
{
    using namespace clipp;
    using namespace std::chrono;

    std::string graph_file;
    bool        validate    = false;
    bool        sym         = false;
    uint32_t    num_runs    = 1;
    uint32_t    num_sources = 0;
    uint32_t    group_size  = 0;
    uint32_t    levels_per_submit = 8;

    auto cli = group(
        value("graph file", graph_file) % "MTX graph file to process",
        (option("-n", "--num-runs") & value("number of runs", num_runs)) %
            "number of times to run the benchmark",
        (option("-k", "--sources") & value("sources", num_sources)) %
            "number of strided sources (default: largest of 64/32/16/8/4 that fits)",
        (option("-g", "--group") & value("sources per dispatch", group_size)) %
            "sources sharing one frontier and dispatch (default: all of -k)",
        (option("-b", "--batch") & value("levels per submit", levels_per_submit)) %
            "forward levels per host submission (1 = one fence wait per level)",
        option("-v", "--validate").set(validate) %
            "validate GPU dependency values against a CPU Brandes reference",
        option("-s", "--sym").set(sym) %
            "symmetric view: import symmetrized (each undirected edge twice)");

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
    auto g = pictor::graphs::Graph::import(ctx, graph_file, sym, pictor::graphs::GraphLayout::CSR);
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

    std::cout << std::format("Graph: {} vertices, {} edges ({} view)\n", g.num_vertices, g.num_edges,
                             g.symmetrized ? "symmetric" : "directed");
    std::cout << std::format("Device pool: {:.2f} GiB\n", free_mem / 1073741824.0);

    if (num_sources == 0)
        num_sources = pictor::graphs::BC::choose_sources(g.num_vertices, free_mem);
    if (group_size == 0)
        group_size = num_sources;
    std::cout << std::format("Sources: {} (strided, i * V / k), {} per fused dispatch\n",
                             num_sources, group_size);

    pictor::graphs::BC bc(ctx, free_mem);

    double   total_ms = 0.0;
    double   fwd_ms   = 0.0;
    double   bwd_ms   = 0.0;
    double   init_ms  = 0.0;
    uint32_t levels   = 0;
    const uint64_t checksum_offset =
        g.num_vertices > cChecksumWindow ? (g.num_vertices - cChecksumWindow) / 2 : 0;
    const uint64_t checksum_count = std::min<uint64_t>(cChecksumWindow, g.num_vertices);

    std::cout << std::format("Running BC {} times...\n", num_runs);
    for (uint32_t run = 0; run < num_runs; ++run) {
        const auto t1 = high_resolution_clock::now();
        levels        = bc.run(g, num_sources, group_size, levels_per_submit);
        const auto t2 = high_resolution_clock::now();

        const double ms = duration<double, std::milli>(t2 - t1).count();

        // Checksum over a fixed window of every source (raw delta, unnormalized).
        const std::vector<float> sample = bc.copy_delta(checksum_offset, checksum_count);
        double                   checksum = 0.0;
        for (float v : sample)
            checksum += v;

        total_ms += ms;
        fwd_ms += bc.forward_ms();
        bwd_ms += bc.backward_ms();
        init_ms += bc.init_ms();

        std::cout << std::format(
            "Run {} - BC Total: {:.3f} ms (init {:.3f}, forward {:.3f}, backward {:.3f}, {} "
            "levels, checksum {:.6f})\n",
            run + 1, ms, bc.init_ms(), bc.forward_ms(), bc.backward_ms(), levels, checksum);
        ctx.wait_idle();
    }

    std::cout << std::format("\n=== Average Results ({} runs) ===\n", num_runs);
    std::cout << std::format("  Average BC total:    {:.3f} ms\n", total_ms / num_runs);
    std::cout << std::format("  Average BC init:     {:.3f} ms\n", init_ms / num_runs);
    std::cout << std::format("  Average BC forward:  {:.3f} ms\n", fwd_ms / num_runs);
    std::cout << std::format("  Average BC backward: {:.3f} ms\n", bwd_ms / num_runs);
    std::cout << std::format("  Levels:              {}\n", levels);
    std::cout << std::format("  Sources:             {} (group {}, {} levels per submit)\n",
                             bc.num_sources(), bc.group_size(), levels_per_submit);
    std::cout << std::format("  Frontier entries:    {} of {} ({:.2f} GiB records, {:.2f} GiB "
                             "frontier)\n",
                             bc.frontier_peak(),
                             static_cast<uint64_t>(g.num_vertices) * bc.group_size(),
                             static_cast<double>(g.num_vertices) * bc.num_sources() * 16.0 /
                                 1073741824.0,
                             static_cast<double>(g.num_vertices) * bc.group_size() * 8.0 /
                                 1073741824.0);

    if (validate) {
        std::cout << "Running CPU Validation..." << std::endl;
        const uint64_t window_offset =
            g.num_vertices > cValidationWindow ? (g.num_vertices - cValidationWindow) / 3 : 0;
        const uint64_t window_count = std::min<uint64_t>(cValidationWindow, g.num_vertices);

        const std::vector<float>    gpu      = bc.copy_delta(window_offset, window_count);
        const std::vector<uint32_t> sources  = bc.source_list();
        pictor::graphs::BCValidator validator(g.row_offsets, g.column_indices);

        bool     ok      = true;
        uint64_t checked = 0;
        double   max_rel = 0.0;

        for (uint32_t s = 0; s < bc.num_sources(); ++s) {
            const auto val_t0 = high_resolution_clock::now();
            const std::vector<float> row(gpu.begin() + static_cast<std::ptrdiff_t>(s) * window_count,
                                         gpu.begin() + static_cast<std::ptrdiff_t>(s + 1) * window_count);
            const auto               cmp =
                validator.compare(sources[s], row, window_offset, 1e-3);

            const double val_ms =
                duration<double, std::milli>(high_resolution_clock::now() - val_t0).count();
            std::cout << std::format(
                "  source {:>3} (v {}): checked {} - max_rel {:.3e}, max_abs {:.3e}, "
                "over tol {}, sum gpu {:.6f} / cpu {:.6f}, sigma max {:.3e} ({:.0f} ms)\n",
                s, sources[s], cmp.checked, cmp.max_rel, cmp.max_abs, cmp.exceeding, cmp.sum_gpu,
                cmp.sum_cpu, validator.sigma_max(), val_ms);
            checked += cmp.checked;
            max_rel = std::max(max_rel, cmp.max_rel);
            ok      = ok && (cmp.exceeding == 0);
        }

        std::cout << std::format("CPU reference: {} values compared over {} sources, max relative "
                                 "deviation {:.3e} (tolerance 1e-3)\n",
                                 checked, bc.num_sources(), max_rel);
        std::cout << (ok ? "SUCCESS: GPU and CPU dependency values match within tolerance.\n"
                         : "FAILURE: GPU and CPU dependency values differ beyond tolerance.\n");
        if (!ok)
            return 2;
    }

    return 0;
}
