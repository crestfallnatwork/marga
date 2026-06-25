// SPDX-License-Identifier: GPL-2.0-only
// Author: Himanshu Gupta <himanshu3@iisc.ac.in>

#include "pagerank.hpp"
#include "pictor_core/context/context.hpp"

#include <chrono>
#include <cstdint>

namespace pictor {
namespace graphs {

PageRank::PageRank(Context &ctx) : _ctx(ctx) {
  reduction.reset(
      pr_shader::ReduceComputeExecutor::create(_ctx).val().release());
  pagerank.reset(pagerank_shader::PRExecutor::create(_ctx).val().release());
  pagerankCompute.reset(
      pagerank_shader::PageComputeExecutor::create(_ctx).val().release());
  prc.reset(PagerankCSCExecutor::create(_ctx).val().release());
  outdegree.reset(OutDegreeExecutor::create(_ctx).val().release());
  dispatch.reset(DispatchExecutor::create(_ctx).val().release());
}

PageRank::~PageRank() { _cleanup_buffers(); }

void PageRank::_allocate_buffers(uint64_t num_vertices, uint64_t num_edges) {
  not_done_flag = _ctx.pool_allocator->allocate(sizeof(ReductionFlag), 0);
  past_pr = _ctx.pool_allocator->allocate(sizeof(float) * num_vertices, 0);
  current_pr = _ctx.pool_allocator->allocate(sizeof(float) * num_vertices, 0);
  _h_pr_results =
      _ctx.pinned_allocator->allocate(sizeof(float) * num_vertices, 0);
  indirect_mesh_grid = _ctx.pool_allocator->allocate(sizeof(uint32_t) * 3, 0);
  indirect_reduce_grid = _ctx.pool_allocator->allocate(sizeof(uint32_t) * 3, 0);
  h_not_done = _ctx.pinned_allocator->allocate(sizeof(ReductionFlag), 0);

  float init_pr = 1. / num_vertices;
  uint32_t punned_pr;
  std::memcpy(&punned_pr, &init_pr, sizeof(float));
  _ctx.fill(past_pr, punned_pr);
  _ctx.fill(current_pr, 0);
  _ctx.fill(not_done_flag, 0);
  _ctx.fill(not_done_flag, sizeof(ReductionFlag) - sizeof(uint32_t), sizeof(uint32_t), 1);
}

void PageRank::_cleanup_buffers() {
  _ctx.deallocate(not_done_flag);
  _ctx.deallocate(past_pr);
  _ctx.deallocate(current_pr);
  _ctx.deallocate(_h_pr_results);
  _ctx.deallocate(out_degree);
  _ctx.deallocate(indirect_mesh_grid);
  _ctx.deallocate(indirect_reduce_grid);
  _ctx.deallocate(h_not_done);
}

std::pair<std::vector<float>, PageRankTimings>
PageRank::run(Graph &G, float delta, GraphLayout layout) {
  using namespace std::chrono_literals;
  using namespace std::chrono;
  /* Prequel allocate buffers and zero them */
  _allocate_buffers(G.num_vertices, G.num_edges);
  _ctx.fsync();

  if (layout == GraphLayout::CSC) {
    auto out_t1 = high_resolution_clock::now();
    out_degree =
        _ctx.pool_allocator->allocate(sizeof(uint32_t) * G.num_vertices, 0);
    _ctx.fill(out_degree, 0);
    _ctx.barrier(out_degree, Access::TransferWrite, Access::Read);
    outdegree->record(
        G.num_vertices, 1, 1,
        {
            .row_offsets =
                reinterpret_cast<uint64_t *>(G.d_row_offsets.address),
            .column_indices =
                reinterpret_cast<uint32_t *>(G.d_column_indices.address),
            .out_degrees = reinterpret_cast<uint32_t *>(out_degree.address),
            .vertices_n = static_cast<uint32_t>(G.num_vertices),
        });
    _ctx.fsync();
    auto out_t2 = high_resolution_clock::now();
    std::cout << std::format("Out Degree Kernel: {}\n",
                             duration<double, std::milli>(out_t2 - out_t1));
  }

  auto times = 0.ms;
  uint64_t iterations = 0;
  volatile ReductionFlag *not_done =
      reinterpret_cast<ReductionFlag *>(h_not_done.mappedPtr);
  auto curr = &current_pr;
  auto past = &past_pr;

  uint batchSize = 10;

  auto t1 = std::chrono::high_resolution_clock::now();
  do {
    for (uint32_t i = 0; i < batchSize; i++) {
      dispatch->record(
          1, 1, 1,
          {
              .flag = reinterpret_cast<ReductionFlag *>(not_done_flag.address),
              .mesh_grid =
                  reinterpret_cast<uint32_t *>(indirect_mesh_grid.address),
              .reduce_grid =
                  reinterpret_cast<uint32_t *>(indirect_reduce_grid.address),
              .vertices_n = static_cast<uint32_t>(G.num_vertices),
          });

      _ctx.barrier(indirect_mesh_grid, Access::Write, Access::IndirectRead); 
      _ctx.barrier(indirect_reduce_grid, Access::Write, Access::IndirectRead);

      _ctx.barrier(*past, Access::Write, Access::Read);
      _ctx.barrier(*curr, Access::Write, Access::Write);

      prc->record_indirect(
          indirect_mesh_grid,
          {
              .row_offsets =
                  reinterpret_cast<uint64_t *>(G.d_row_offsets.address),
              .column_indices =
                  reinterpret_cast<uint32_t *>(G.d_column_indices.address),
              .past_pr = reinterpret_cast<float *>(past->address),
              .curr_pr = reinterpret_cast<float *>(curr->address),
              .out_degrees = reinterpret_cast<uint32_t *>(out_degree.address),
              .vertices_n = G.num_vertices,
              .edges_n = G.num_edges,
          });

      _ctx.barrier(*curr, Access::Write, Access::Read);
      _ctx.barrier(not_done_flag, Access::Write, Access::Read);
      _ctx.barrier(not_done_flag, Access::Write, Access::Write);

      ReductionIn reductionIn = {
          .past_pr = reinterpret_cast<float *>(past->address),
          .current_pr = reinterpret_cast<float *>(curr->address),
          .flag = reinterpret_cast<ReductionFlag *>(not_done_flag.address),
          .delta = delta,
          .vertices_n = G.num_vertices,
          .d = 0.85};
      reduction->record_indirect(indirect_reduce_grid, reductionIn);

      _ctx.barrier(not_done_flag, Access::Write, Access::Read);

      std::swap(curr, past);
    }
    _ctx.barrier(not_done_flag, Access::Write, Access::TransferRead);
    _ctx.copy(not_done_flag, h_not_done);
    _ctx.fsync();

    _ctx.invalidate(h_not_done);
    iterations++;
  } while (not_done->flag);
  auto t2 = std::chrono::high_resolution_clock::now();
  times = t2 - t1;
  iterations = (not_done->iterations != 0) ? not_done->iterations : iterations;
  std::cout << std::format("\nIterations approx: {}, Time: {}, Time/Iteration(approx): {}",
                           iterations, times, times/iterations)
            << std::endl;
  past = (iterations % 2) ? &current_pr : &past_pr;
  _ctx.copy(*past, _h_pr_results);
  _ctx.fsync();

  std::vector<float> pr;
  float *pr_results = reinterpret_cast<float *>(_h_pr_results.mappedPtr);
  _ctx.invalidate(_h_pr_results);
  for (int i = 0; i < G.num_vertices; i++) {
    pr.push_back(pr_results[i]);
  }
  std::cout << std::format("Pageranks: [1] {}, [2] {}, [3] {}, [4] {}",
                           pr_results[0], pr_results[1], pr_results[2],
                           pr_results[3])
            << std::endl;
  return {pr, {}};
}

} // namespace graphs
} // namespace pictor
