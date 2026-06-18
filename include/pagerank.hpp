#pragma once

#include "csr.hpp"
#include "pictor_core/context/context.hpp"

#include "pagerank.shader.h"

#include <vector>

namespace pictor {
namespace graphs {

struct PageRankTimings {
  double init_ms = 0;
  double preprocessing_ms = 0;
  double iteration_avg_ms = 0;
  double total_ms = 0;
  uint32_t num_iterations = 0;
};

class PageRank {
public:
  PageRank(Context &ctx);
  ~PageRank();

  std::pair<std::vector<float>, PageRankTimings> run(Graph &g, float delta);

private:
  Context &_ctx;

  std::unique_ptr<pagerank_shader::ReduceComputeExecutor> reduction;
  using ReductionFlag = pagerank_shader::ReductionFlag;
  using ReductionIn = pagerank_shader::ReductionIn;

  
  std::unique_ptr<pagerank_shader::PageComputeExecutor> pagerankCompute;
  std::unique_ptr<pagerank_shader::PRExecutor> pagerank;
  using PageRankIn = pagerank_shader::PageRankIn;

  AllocationInfo not_done_flag;
  AllocationInfo past_pr;
  AllocationInfo current_pr;
  AllocationInfo _h_pr_results;

  void _allocate_buffers(uint64_t num_vertices, uint64_t num_edges);
  void _cleanup_buffers();
};

} // namespace graphs
} // namespace pictor
