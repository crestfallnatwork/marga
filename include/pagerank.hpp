#pragma once

#include "csr.hpp"
#include "pictor_core/context/context.hpp"
#include "pictor_core/execution/compute_executor.hpp"
#include "pictor_core/execution/mesh_executor.hpp"
#include "out_degree.layout.hpp"
#include "pagerank.layout.hpp"

#include <memory>
#include <vector>
#include <span>

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
    PageRank(Context& ctx);
    ~PageRank();

    void preprocess(const Graph& graph);
    std::pair<std::vector<float>, PageRankTimings> run(const Graph& graph, uint32_t max_iterations = 20);

private:
    Context& _ctx;
    
    std::unique_ptr<ComputeExecutor<out_degree_json_layout::Params>> _out_degree_exec;
    
    std::unique_ptr<MeshExecutor<pagerank_json_layout::Params>> _pr_mesh_exec;
    std::unique_ptr<ComputeExecutor<pagerank_json_layout::Params>> _pr_reduce_exec;
    std::unique_ptr<ComputeExecutor<pagerank_json_layout::Params>> _pr_finalize_exec;
    std::unique_ptr<ComputeExecutor<pagerank_json_layout::Params>> _pr_dangling_exec;

    AllocationInfo _out_degrees;
    AllocationInfo _pr_current;
    AllocationInfo _pr_next;
    AllocationInfo _active_vertices_curr;
    AllocationInfo _active_count_curr;
    AllocationInfo _dangling_sum;
    AllocationInfo _intermediate;
    AllocationInfo _intermediate_count;
    
    // For reading back results
    AllocationInfo _h_pr_results;

    void _allocate_buffers(uint64_t num_vertices, uint64_t num_edges);
    void _cleanup_buffers();
};

} // namespace graphs
} // namespace pictor
