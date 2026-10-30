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

    // Samples `sources` strided sources (i * V / sources). `group` sources
    // share one frontier and one dispatch per level; all sources are processed
    // either way. Returns the number of levels of the deepest group.
    uint32_t run(Graph const& g, uint32_t sources, uint32_t group);

    // Largest source count in {4, 8, 16, 32, 64} whose records and frontier
    // fit in 60% of `pool_bytes`.
    static uint32_t choose_sources(uint64_t num_vertices, size_t pool_bytes);

    // Raw dependency sums (delta) for `count` vertices starting at
    // `vertex_offset`, one row of `count` floats per source. No normalization.
    std::vector<float> copy_delta(uint64_t vertex_offset, uint64_t count);

    // Source vertices actually sampled, in order.
    std::vector<uint32_t> source_list() const;

    double   init_ms() const { return init_ms_; }
    double   forward_ms() const { return forward_ms_; }
    double   backward_ms() const { return backward_ms_; }
    uint32_t levels() const { return levels_; }
    uint32_t num_sources() const { return num_sources_; }
    uint32_t group_size() const { return group_; }
    uint64_t vertex_count() const { return num_vertices_; }
    uint64_t frontier_peak() const { return frontier_peak_; }

    using FwdExec     = bc_shader::BC_FWDExecutor;
    using FwdIn       = bc_shader::BCFwdIn;
    using BwdExec     = bc_shader::BwdComputeExecutor;
    using BwdIn       = bc_shader::BCBwdIn;
    using PrepareExec = bc_shader::PrepareComputeExecutor;
    using PrepareIn   = bc_shader::BCPrepareIn;

  private:
    void release();
    void allocate(uint64_t num_vertices, uint32_t sources, uint32_t group);
    void prepare(uint32_t sources);
    void seed_frontier(uint32_t first_source, uint32_t last_source);
    uint32_t forward(Graph const& g, uint32_t first_source, uint32_t last_source,
                     std::vector<uint64_t>& level_base, std::vector<uint64_t>& level_count);
    void     backward(Graph const& g, std::vector<uint64_t> const& level_base,
                      std::vector<uint64_t> const& level_count);

    Context& ctx;
    size_t   pool_bytes_ = 0;

    std::unique_ptr<FwdExec>     fwd_exec;
    std::unique_ptr<BwdExec>     bwd_exec;
    std::unique_ptr<PrepareExec> prepare_exec;

    AllocationInfo d_records{};
    AllocationInfo d_frontier{};
    AllocationInfo d_counters{};
    AllocationInfo h_counters{};
    AllocationInfo h_delta{};

    uint64_t num_vertices_   = 0;
    uint64_t frontier_cap_   = 0;
    uint64_t frontier_peak_  = 0;
    uint32_t num_sources_    = 0;
    uint32_t group_          = 0;
    uint32_t levels_         = 0;
    double   init_ms_        = 0.0;
    double   forward_ms_     = 0.0;
    double   backward_ms_    = 0.0;
};

} // namespace graphs
} // namespace pictor
