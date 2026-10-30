// SPDX-License-Identifier: GPL-2.0-only
// Author: Himanshu Gupta <himanshu3@iisc.ac.in>

#pragma once

#include "cc.shader.h"
#include "pictor_core/memory/allocation_type.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace pictor {
struct Context;

namespace graphs {
struct Graph;

// Connected components by Shiloach-Vishkin-style hooking over a shrinking
// active edge list: the task stage culls resolved edges and compacts the live
// ones, the mesh stage performs one conditional root hook per live edge.
class CC {
  public:
    explicit CC(Context& ctx);
    ~CC();

    // Runs the round loop and returns the number of rounds executed. With
    // `symmetric_view` the task stage processes each undirected edge once
    // (culls u > v); the graph must then be imported symmetrized.
    uint32_t run(Graph const& g, bool symmetric_view);

    // Root of every vertex after convergence (path-compressed on the host).
    // The parent array stays on the device between calls; this copies it back.
    std::vector<uint32_t> resolve_roots();

    uint64_t component_count();

    double init_ms() const { return init_ms_; }
    double loop_ms() const { return loop_ms_; }
    uint32_t rounds() const { return rounds_; }

    using CCExec      = cc_shader::CCExecutor;
    using CCIn        = cc_shader::CCIn;
    using PrepareExec = cc_shader::PrepareComputeExecutor;
    using PrepareIn   = cc_shader::PrepareIn;

  private:
    void prepare(Graph const& g);
    void release();

    Context& ctx;
    std::unique_ptr<CCExec> exec;
    std::unique_ptr<PrepareExec> prepare_exec;

    AllocationInfo d_parent{};
    AllocationInfo d_edge_src{};
    AllocationInfo d_active[2]{};
    AllocationInfo d_counters{};
    AllocationInfo h_counters{};
    AllocationInfo h_parent{};

    uint64_t num_vertices_ = 0;
    uint64_t num_edges_    = 0;
    uint32_t rounds_       = 0;
    double init_ms_        = 0.0;
    double loop_ms_        = 0.0;
};

} // namespace graphs
} // namespace pictor
