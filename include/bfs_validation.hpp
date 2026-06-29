// SPDX-License-Identifier: GPL-2.0-only
// Author: Himanshu Gupta <himanshu3@iisc.ac.in>

#pragma once

/*
 * Adapted from GAP Benchmark Suite (Scott Beamer)
 * Adapted by Gemini CLI
 */

#include "csr.hpp"
#include <cstdint>
#include <span>
#include <vector>

namespace pictor {
namespace graphs {

struct ValidationTimings {
    double traversal_ms = 0;
};

class BFSValidator {
  public:
    static std::pair<std::vector<uint32_t>, ValidationTimings> run(const Graph& graph,
                                                                   uint32_t source);

    static bool verify(const Graph& graph, uint32_t source, std::span<const uint32_t> gpu_distances,
                       std::span<const uint32_t> cpu_distances);
};

} // namespace graphs
} // namespace pictor
