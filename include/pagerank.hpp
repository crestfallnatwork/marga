// SPDX-License-Identifier: GPL-2.0-only
// Author: Himanshu Gupta <himanshu3@iisc.ac.in>

#pragma once

#include "csr.hpp"
#include "pictor_core/context/context.hpp"

#include "pr.shader.h"

#include <vector>

namespace pictor {
namespace graphs {

class PageRank {
  public:
    PageRank(Context& ctx);
    ~PageRank();

    void compute_out_degree(Graph& g, GraphLayout layout = GraphLayout::CSR);
    uint32_t run(Graph& g, float delta);
    std::vector<float> copy_to_host(uint32_t num_vertices);

  private:
    Context& _ctx;

    std::unique_ptr<pr_shader::ReduceComputeExecutor> reduction;
    using ReductionIn = pr_shader::ReductionIn;

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
    
    AllocationInfo* _final_pr = nullptr;

    void _allocate_buffers(uint64_t num_vertices, uint64_t num_edges);
    void _cleanup_buffers();
};

} // namespace graphs
} // namespace pictor
