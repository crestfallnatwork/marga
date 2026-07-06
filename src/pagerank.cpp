// SPDX-License-Identifier: GPL-2.0-only
// Author: Himanshu Gupta <himanshu3@iisc.ac.in>

#include "pagerank.hpp"
#include "helper.hpp"

#include "pictor_core/context/context.hpp"

#include <chrono>
#include <cstdint>

namespace pictor {
namespace graphs {

PageRank::PageRank(Context& ctx) : _ctx(ctx)
{
    reduction.reset(pr_shader::ReduceComputeExecutor::create(_ctx).val().release());
    prc.reset(PagerankCSCExecutor::create(_ctx).val().release());
    outdegree.reset(OutDegreeExecutor::create(_ctx).val().release());
    dispatch.reset(DispatchExecutor::create(_ctx).val().release());
    danglingHandler.reset(DanglingHandler::create(_ctx).val().release());
}

PageRank::~PageRank()
{
    _cleanup_buffers();
}

void PageRank::_allocate_buffers(uint64_t num_vertices, uint64_t num_edges)
{
    CheckAllocation(past_pr = _ctx.pool_allocator->allocate(sizeof(float) * (num_vertices + 1), 0));
    CheckAllocation(current_pr =
                        _ctx.pool_allocator->allocate(sizeof(float) * (num_vertices + 1), 0));
    CheckAllocation(_h_pr_results =
                        _ctx.pinned_allocator->allocate(sizeof(float) * num_vertices, 0));
    CheckAllocation(indirect_mesh_grid = _ctx.pool_allocator->allocate(sizeof(uint32_t) * 3, 0));
    CheckAllocation(indirect_reduce_grid = _ctx.pool_allocator->allocate(sizeof(uint32_t) * 3, 0));
    CheckAllocation(_d_delta = _ctx.pool_allocator->allocate(sizeof(float), 0));
    CheckAllocation(_iterations = _ctx.pool_allocator->allocate(sizeof(uint32_t), 0));

    float init_pr = 1. / num_vertices;
    uint32_t punned_pr;
    std::memcpy(&punned_pr, &init_pr, sizeof(float));
    _ctx.fill(past_pr, punned_pr);
    _ctx.fill(current_pr, 0);
    uint32_t punned;
    float one = 1.;
    std::memcpy(&punned, &one, sizeof(float));
    _ctx.fill(_d_delta, punned);
    _ctx.fill(_iterations, 0);
}

void PageRank::_cleanup_buffers()
{
    _ctx.deallocate(past_pr);
    _ctx.deallocate(current_pr);
    _ctx.deallocate(_h_pr_results);
    _ctx.deallocate(out_degree);
    _ctx.deallocate(indirect_mesh_grid);
    _ctx.deallocate(indirect_reduce_grid);
    _ctx.deallocate(_d_delta);
    _ctx.deallocate(_iterations);
}

void PageRank::compute_out_degree(Graph& G, GraphLayout layout)
{
    using namespace std::chrono_literals;
    using namespace std::chrono;
    if (layout == GraphLayout::CSC) {
        auto out_t1 = high_resolution_clock::now();
        out_degree  = _ctx.pool_allocator->allocate(sizeof(uint32_t) * G.num_vertices, 0);
        _ctx.fill(out_degree, 0);
        _ctx.barrier(out_degree, Access::TransferWrite, Access::Write);
        outdegree->record(
            G.num_vertices, 1, 1,
            {
                .row_offsets    = reinterpret_cast<uint64_t*>(G.d_row_offsets.address),
                .column_indices = reinterpret_cast<uint32_t*>(G.d_column_indices.address),
                .out_degrees    = reinterpret_cast<uint32_t*>(out_degree.address),
                .vertices_n     = static_cast<uint32_t>(G.num_vertices),
            });
        _ctx.fsync();
    }
}

uint32_t PageRank::run(Graph& G, float delta)
{
    /* Prequel allocate buffers and zero them */
    _allocate_buffers(G.num_vertices, G.num_edges);
    _ctx.fsync();

    auto curr  = &current_pr;
    auto past  = &past_pr;

    uint batchSize = 25;

    volatile float* h_delta = reinterpret_cast<float*>(_h_pr_results.mappedPtr);

    auto t1 = std::chrono::high_resolution_clock::now();
    uint32_t current_iter = 0;
    do {
        for (uint32_t i = 0; i < batchSize; i++) {
            _ctx.barrier(_iterations, Access::Write, Access::Write);
            _ctx.barrier(*curr, Access::Write, Access::Write);
            dispatch->record(
                1, 1, 1,
                {
                    .d_delta     = reinterpret_cast<float*>(_d_delta.address),
                    .curr_pr     = reinterpret_cast<float*>(curr->address),
                    .iterations  = reinterpret_cast<uint32_t*>(_iterations.address),
                    .mesh_grid   = reinterpret_cast<uint32_t*>(indirect_mesh_grid.address),
                    .reduce_grid = reinterpret_cast<uint32_t*>(indirect_reduce_grid.address),
                    .vertices_n  = static_cast<uint32_t>(G.num_vertices),
                    .delta       = delta,
                });

            _ctx.barrier(*past, Access::Write, Access::Read);

            _ctx.barrier(indirect_mesh_grid, Access::Write, Access::IndirectRead);
            prc->record_indirect(
                indirect_mesh_grid,
                {
                    .row_offsets    = reinterpret_cast<uint64_t*>(G.d_row_offsets.address),
                    .column_indices = reinterpret_cast<uint32_t*>(G.d_column_indices.address),
                    .past_pr        = reinterpret_cast<float*>(past->address),
                    .curr_pr        = reinterpret_cast<float*>(curr->address),
                    .out_degrees    = reinterpret_cast<uint32_t*>(out_degree.address),
                    .vertices_n     = G.num_vertices,
                    .edges_n        = G.num_edges,
                    .iter           = current_iter,
                });

            _ctx.barrier(indirect_reduce_grid, Access::Write, Access::IndirectRead);
            danglingHandler->record_indirect(
                indirect_reduce_grid,
                {
                    .past_pr    = reinterpret_cast<float*>(past->address),
                    .curr_pr    = reinterpret_cast<float*>(curr->address),
                    .out_degree = reinterpret_cast<uint32_t*>(out_degree.address),
                    .vertices_n = G.num_vertices,
                });

            _ctx.barrier(*curr, Access::Write, Access::Read);
            _ctx.barrier(_d_delta, Access::Write, Access::Write);

            ReductionIn reductionIn = {
                .past_pr         = reinterpret_cast<float*>(past->address),
                .current_pr      = reinterpret_cast<float*>(curr->address),
                .d_delta         = reinterpret_cast<float*>(_d_delta.address),
                .delta           = delta,
                .d               = 0.85,
                .vertices_n      = G.num_vertices,
                .out_degrees     = reinterpret_cast<uint32_t*>(out_degree.address),
                .is_restore_mode = 0,
            };
            reduction->record_indirect(indirect_reduce_grid, reductionIn);

            _ctx.barrier(_d_delta, Access::Write, Access::Read);

            std::swap(curr, past);
            current_iter++;
        }
        _ctx.barrier(_d_delta, Access::Write, Access::TransferRead);
        _ctx.copy(_d_delta, _h_pr_results);
        _ctx.fsync();
        _ctx.invalidate(_h_pr_results);
    } while (*h_delta > delta);

    _ctx.copy(_iterations, _h_pr_results);
    _ctx.fsync();
    _ctx.invalidate(_h_pr_results);
    volatile auto h_iterations = reinterpret_cast<uint32_t*>(_h_pr_results.mappedPtr);
    uint32_t iterations        = *h_iterations;

    past = (iterations % 2) ? &current_pr : &past_pr;

    // Final restoration kernel to multiply by out_degrees
    _ctx.barrier(*past, Access::Write, Access::Write);
    ReductionIn finalReductionIn = {
        .past_pr         = reinterpret_cast<float*>(past->address),
        .current_pr      = reinterpret_cast<float*>(past->address),    // unused
        .d_delta         = reinterpret_cast<float*>(_d_delta.address), // unused
        .delta           = delta,
        .d               = 0.85,
        .vertices_n      = G.num_vertices,
        .out_degrees     = reinterpret_cast<uint32_t*>(out_degree.address),
        .is_restore_mode = 1,
    };
    reduction->record(G.num_vertices, 1, 1, finalReductionIn);
    _ctx.barrier(*past, Access::Write, Access::TransferRead);
    _ctx.fsync();

    _final_pr = past;
    
    return iterations;
}

std::vector<float> PageRank::copy_to_host(uint32_t num_vertices)
{
    _ctx.copy(*_final_pr, _h_pr_results);
    _ctx.fsync();

    float* pr_results = reinterpret_cast<float*>(_h_pr_results.mappedPtr);
    _ctx.invalidate(_h_pr_results);
    std::vector<float> pr(pr_results, pr_results + num_vertices);
    return pr;
}

} // namespace graphs
} // namespace pictor
