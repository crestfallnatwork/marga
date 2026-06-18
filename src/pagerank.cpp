#include "pagerank.hpp"
#include "pictor_core/context/context.hpp"

#include <chrono>
#include <cstdint>

namespace pictor {
namespace graphs {

PageRank::PageRank(Context &ctx) : _ctx(ctx) {
  reduction.reset(
      pagerank_shader::ReduceComputeExecutor::create(_ctx).val().release());
  pagerank.reset(pagerank_shader::PRExecutor::create(_ctx).val().release());
  pagerankCompute.reset(pagerank_shader::PageComputeExecutor::create(_ctx).val().release());
}

PageRank::~PageRank() { _cleanup_buffers(); }

void PageRank::_allocate_buffers(uint64_t num_vertices, uint64_t num_edges) {
  not_done_flag = _ctx.pinned_allocator->allocate(sizeof(ReductionFlag), 0);
  past_pr = _ctx.pool_allocator->allocate(sizeof(float) * num_vertices, 0);
  current_pr = _ctx.pool_allocator->allocate(sizeof(float) * num_vertices, 0);
  _h_pr_results =
      _ctx.pinned_allocator->allocate(sizeof(float) * num_vertices, 0);
  float init_pr = 1. / num_vertices;
  uint32_t punned_pr;
  std::memcpy(&punned_pr, &init_pr, sizeof(float));
  _ctx.fill(past_pr, punned_pr);
  _ctx.fill(current_pr, 0);
  _ctx.fill(not_done_flag, 1);
}

void PageRank::_cleanup_buffers() {
  _ctx.deallocate(not_done_flag);
  _ctx.deallocate(past_pr);
  _ctx.deallocate(current_pr);
  _ctx.deallocate(_h_pr_results);
}

std::pair<std::vector<float>, PageRankTimings> PageRank::run(Graph &G,
                                                             float delta) {
  using namespace std::chrono_literals;
  using namespace std::chrono;
  /* Prequel allocate buffers and zero them */
  _allocate_buffers(G.num_vertices, G.num_edges);
  _ctx.fsync();

  auto times = 0ms;
  uint64_t iterations = 0;
  volatile ReductionFlag *h_not_done =
      reinterpret_cast<ReductionFlag *>(not_done_flag.mappedPtr);
  auto curr = &current_pr;
  auto past = &past_pr;
  while (h_not_done->flag) {
    h_not_done->flag = false;
    _ctx.flush(not_done_flag);
    auto t1 = std::chrono::high_resolution_clock::now();
    PageRankIn pagerankIn = {
        .row_offsets = reinterpret_cast<uint64_t *>(G.d_row_offsets.address),
        .column_indices =
            reinterpret_cast<uint32_t *>(G.d_column_indices.address),
        .current_pr = reinterpret_cast<float *>(curr->address),
        .past_pr = reinterpret_cast<float *>(past->address),
        .vertices_n = G.num_vertices,
        .edges_n = G.num_edges};
    pagerank->record(G.num_vertices, 1, 1, pagerankIn);
    // pagerankCompute->record(G.num_edges, 1, 1, pagerankIn);

    _ctx.barrier(*curr, Access::Write, Access::Read);

    ReductionIn reductionIn = {
        .past_pr = reinterpret_cast<float *>(past->address),
        .current_pr = reinterpret_cast<float *>(curr->address),
        .flag = reinterpret_cast<ReductionFlag *>(not_done_flag.address),
        .delta = delta,
        .vertices_n = G.num_vertices,
        .d = 0.85};
    reduction->record(G.num_vertices, 1, 1, reductionIn);

    _ctx.fsync();
    auto t2 = std::chrono::high_resolution_clock::now();

    std::swap(curr, past);

    _ctx.invalidate(not_done_flag);

    std::cout << std::format("Iteration: {}, Time: {}\n", iterations, duration_cast<milliseconds>(t2 - t1));

    times = times + duration_cast<milliseconds>(t2 - t1);
    iterations++;
  }
  std::cout << std::format("Iterations: {}, Time: {}, Time/Iteration: {}",
                           iterations, times, times / iterations)
            << std::endl;
  _ctx.copy(*past, _h_pr_results);
  _ctx.fsync();

  float *pr_results = reinterpret_cast<float *>(_h_pr_results.mappedPtr);
  _ctx.invalidate(_h_pr_results);
  std::cout << std::format("Pageranks: [1] {}, [2] {}, [3] {}, [4] {}",
                           pr_results[0], pr_results[1], pr_results[2],
                           pr_results[3])
            << std::endl;
  return {};
}

} // namespace graphs
} // namespace pictor
