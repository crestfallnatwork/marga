#pragma once

#include "csr.hpp"
#include <vector>
#include <cstdint>
#include <span>

namespace pictor {
namespace graphs {

struct PRValidationTimings {
    double traversal_ms = 0;
};

class PageRankValidator {
public:
    // Runs a parallel CPU PageRank using the Pull model (requires CSC)
    static std::pair<std::vector<float>, PRValidationTimings> run(const Graph& graph, uint32_t max_iterations = 20, float damping = 0.85f);
    
    // Verifies GPU results against CPU results
    static bool verify(const Graph& graph, 
                      std::span<const float> gpu_results, 
                      std::span<const float> cpu_results,
                      float tolerance = 1e-4f);
};

} // namespace graphs
} // namespace pictor
