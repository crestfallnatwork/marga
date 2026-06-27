// SPDX-License-Identifier: GPL-2.0-only
// Author: Himanshu Gupta <himanshu3@iisc.ac.in>

#pragma once

#include "csr.hpp"
#include "pictor_core/context/context.hpp"

#include "pagerank.shader.h"
#include "pr.shader.h"

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

  std::pair<std::vector<float>, PageRankTimings>
  run(Graph &g, float delta, GraphLayout layout = GraphLayout::CSR);

private:
  Context &_ctx;

  std::unique_ptr<pr_shader::ReduceComputeExecutor> reduction;
  using ReductionIn = pr_shader::ReductionIn;

  std::unique_ptr<pagerank_shader::PageComputeExecutor> pagerankCompute;
  std::unique_ptr<pagerank_shader::PRExecutor> pagerank;
  using PageRankIn = pagerank_shader::PageRankIn;

  using PagerankCSCExecutor = pr_shader::PRCExecutor;
  std::unique_ptr<PagerankCSCExecutor> prc;
  using PRCIn = pr_shader::PageRankCSCIn;

  using OutDegreeExecutor = pr_shader::OutdegreeComputeExecutor;
  std::unique_ptr<OutDegreeExecutor> outdegree;
  using OutDegreeIn = pr_shader::OutDegreeIn;

  using DanglingHandler = pr_shader::DanglingComputeExecutor;
  std::unique_ptr<DanglingHandler> danglingHandler;
  using DanglingHandlerIn = pr_shader::DanglingIn;

  using DispatchExecutor = pr_shader::DispatchComputeExecutor;
  std::unique_ptr<DispatchExecutor> dispatch;
  using DispatchIn = pr_shader::DispatchIn;


  AllocationInfo past_pr;
  AllocationInfo current_pr;
  AllocationInfo out_degree;
  AllocationInfo _h_pr_results;
  AllocationInfo _d_delta;
  AllocationInfo _iterations;
  AllocationInfo indirect_mesh_grid;
  AllocationInfo indirect_reduce_grid;

  void _allocate_buffers(uint64_t num_vertices, uint64_t num_edges);
  void _cleanup_buffers();
};

} // namespace graphs
} // namespace pictor
