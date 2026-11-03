// SPDX-License-Identifier: GPL-2.0-only
// Author: Himanshu Gupta <himanshu3@iisc.ac.in>

#pragma once

#include "kc.shader.h"
#include "pictor_core/memory/allocation_type.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace pictor {
struct Context;

namespace graphs {
struct Graph;

// Core-number decomposition (Batagelj-Zaversnik threshold sweep) over a
// symmetric simple graph. The task stage sizes each cascade wave from the
// frontier vertices' row lengths and the mesh stage decrements one live
// neighbour per out-edge, claiming it when its remaining degree reaches the
// current threshold. A chain of device-side phases advances the threshold and
// partitions the surviving vertices, so the host submits batches of phases
// instead of one dispatch per threshold or per wave.
class KC {
  public:
    explicit KC(Context& ctx);
    ~KC();
    // Runs the full decomposition. `phases_per_submit` phases run per host
    // submission (1 = one fence wait per phase). Returns the number of phases
    // the device ran.
    uint64_t run(Graph const& g, uint32_t phases_per_submit = 32);

    // Core number per vertex, index = vertex id. Only call after run().
    std::vector<uint32_t> copy_core();

    double   init_ms() const { return init_ms_; }
    double   loop_ms() const { return loop_ms_; }
    uint64_t phases() const { return phases_; }
    uint64_t frontier_peak() const { return frontier_peak_; }
    uint32_t max_core() const { return max_core_; }
    uint32_t thresholds() const { return thresholds_; }
    uint64_t vertex_count() const { return num_vertices_; }

    using WorkExec = kc_shader::KCExecutor;
    using WorkIn   = kc_shader::KCIn;
    using InitExec = kc_shader::KcInitComputeExecutor;
    using StepExec = kc_shader::KcStepComputeExecutor;
    using StepIn   = kc_shader::KCStepIn;

  private:
    void release();
    void allocate(uint64_t num_vertices);
    void initialise(Graph const& g);

    Context& ctx;

    std::unique_ptr<WorkExec> work_exec;
    std::unique_ptr<InitExec> init_exec;
    std::unique_ptr<StepExec> step_exec;

    AllocationInfo d_state{};
    AllocationInfo d_counters{};
    AllocationInfo d_grid{};
    AllocationInfo d_deg{};
    AllocationInfo d_core{};
    AllocationInfo d_removed{};
    AllocationInfo d_survivors[2]{};
    AllocationInfo d_frontier{};

    AllocationInfo h_state{};
    AllocationInfo h_counters{};
    AllocationInfo h_core{};

    uint64_t num_vertices_  = 0;
    uint64_t frontier_cap_  = 0;
    uint64_t phases_        = 0;
    uint64_t frontier_peak_ = 0;
    uint32_t max_core_      = 0;
    uint32_t thresholds_    = 0;
    double   init_ms_       = 0.0;
    double   loop_ms_       = 0.0;
};

} // namespace graphs
} // namespace pictor
