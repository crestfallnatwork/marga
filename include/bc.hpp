// SPDX-License-Identifier: GPL-2.0-only
// Author: Himanshu Gupta <himanshu3@iisc.ac.in>

#pragma once

#include "bc.shader.h"
#include "pictor_core/memory/allocation_type.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace pictor {
struct Context;

namespace graphs {
struct Graph;

// Multi-source Brandes betweenness centrality. The forward pass counts shortest
// paths level by level on the mesh pipeline (task stage sizes the frontier's
// out-degree work, mesh stage claims and accumulates sigma); the backward pass
// walks the levels in reverse and gathers each item's dependency from its
// out-edges in one thread.
class BC {
  public:
    // `pool_bytes` is the device pool the driver created for this context.
    explicit BC(Context& ctx, size_t pool_bytes = 0);
    ~BC();

    // Samples `sources` strided sources (i * V / sources) and processes them in
    // internal batches sized to the device pool; `group` is derived from the
    // record set size. `levels_per_submit` forward levels run per host
    // submission (1 = one fence wait per level). `hub_degree` expands
    // dependency items of that out-degree or more over one warp each (0: one
    // thread per item). Returns the levels of the deepest batch.
    uint32_t run(Graph const& g, uint32_t sources, uint32_t levels_per_submit = 8,
                 bool scatter_backward = false, uint32_t hub_degree = 4096);

    // Largest batch in {4, 8, 16, 32, 64} whose records and frontier fit in
    // 60% of `pool_bytes`.
    static uint32_t choose_batch(uint64_t num_vertices, size_t pool_bytes);

    // Raw dependency sums (delta) for `count` vertices starting at
    // `vertex_offset`, one row of `count` floats per source of the batch that
    // was processed last. No normalization.
    std::vector<float> copy_delta(uint64_t vertex_offset, uint64_t count);

    // Source vertices sampled in the batch that was processed last, in order.
    // Sampled sources with no out-edges contribute zero dependency to every
    // vertex (they have no successors to accumulate from) and are dropped
    // before allocation.
    std::vector<uint32_t> source_list() const;

    double   init_ms() const { return init_ms_; }
    double   forward_ms() const { return forward_ms_; }
    double   backward_ms() const { return backward_ms_; }
    uint32_t levels() const { return levels_; }
    uint32_t num_sources() const { return num_sources_; }
    // Sources in the internal batch: the largest set whose state fits the pool.
    uint32_t batch_size() const { return batch_; }
    // Rows with state allocated: the sources of the last batch that have an
    // out-edge.
    uint32_t live_sources() const { return rows_; }
    uint32_t group_size() const { return group_; }
    // Out-degree at which the dependency pass expands an item over one warp
    // (0 when the path is off for this graph).
    uint32_t hub_threshold() const { return hub_threshold_; }
    uint64_t vertex_count() const { return num_vertices_; }
    uint64_t frontier_peak() const { return frontier_peak_; }

    using FwdExec     = bc_shader::BC_FWDExecutor;
    using FwdIn       = bc_shader::BCFwdIn;
    using BwdExec     = bc_shader::BcBwdComputeExecutor;
    using BwdMeshExec = bc_shader::BC_BWDMExecutor;
    using BwdIn       = bc_shader::BcBwdCompute;
    using HubExec     = bc_shader::BcBwdhComputeExecutor;
    using WarpExec    = bc_shader::BcBwdwComputeExecutor;
    using StepExec    = bc_shader::BcStepComputeExecutor;
    using StepIn      = bc_shader::BCStepIn;
    using PrepareExec = bc_shader::BcPrepareComputeExecutor;
    using PrepareIn   = bc_shader::BcPrepareCompute;

  private:
    void release();
    void allocate(uint64_t num_vertices, uint32_t sources, uint32_t group);
    void prepare(uint32_t sources);
    void seed_frontier(uint32_t first_source, uint32_t last_source);
    uint32_t forward(Graph const& g, uint32_t first_source, uint32_t last_source,
                     std::vector<uint64_t>& level_base, std::vector<uint64_t>& level_count);
    void     read_levels(uint64_t recorded, std::vector<uint64_t>& level_base,
                         std::vector<uint64_t>& level_count);
    void     backward(Graph const& g, std::vector<uint64_t> const& level_base,
                      std::vector<uint64_t> const& level_count);
    void     backward_scatter(Graph const& g, std::vector<uint64_t> const& level_base,
                              std::vector<uint64_t> const& level_count);

    Context& ctx;
    size_t   pool_bytes_ = 0;

    std::unique_ptr<FwdExec>     fwd_exec;
    std::unique_ptr<BwdExec>     bwd_exec;
    std::unique_ptr<BwdMeshExec> bwdm_exec;
    std::unique_ptr<HubExec>     hub_exec;
    std::unique_ptr<WarpExec>    warp_exec;
    std::unique_ptr<StepExec>    step_exec;
    std::unique_ptr<PrepareExec> prepare_exec;

    AllocationInfo d_records{};
    AllocationInfo d_frontier{};
    AllocationInfo d_counters{};
    AllocationInfo d_level_state{};
    AllocationInfo d_levels{};
    AllocationInfo d_grid{};
    AllocationInfo d_sources{};
    AllocationInfo d_hub{};
    AllocationInfo d_probe{};
    AllocationInfo h_counters{};
    AllocationInfo h_state{};
    AllocationInfo h_levels{};
    AllocationInfo h_sources{};
    AllocationInfo h_delta{};

    uint64_t num_vertices_   = 0;
    uint64_t frontier_cap_   = 0;
    uint64_t frontier_peak_  = 0;
    uint32_t num_sources_    = 0;
    uint32_t batch_          = 0;
    uint32_t batch_first_    = 0;
    uint32_t rows_           = 0;
    std::vector<uint32_t> live_sources_;
    uint32_t group_          = 0;
    uint32_t levels_         = 0;
    uint32_t levels_per_submit_ = 8;
    uint64_t levels_capacity_   = 0;
    bool     scatter_backward_  = false;
    uint32_t hub_threshold_     = 0;
    double   init_ms_        = 0.0;
    double   forward_ms_     = 0.0;
    double   backward_ms_    = 0.0;
};

} // namespace graphs
} // namespace pictor
