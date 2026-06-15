#include "pagerank_validation.hpp"
#include <chrono>
#include <iostream>
#include <numeric>
#include <cmath>
#include <omp.h>

namespace pictor {
namespace graphs {

std::pair<std::vector<float>, PRValidationTimings> PageRankValidator::run(const Graph& graph, uint32_t max_iterations, float damping) {
    PRValidationTimings timings;
    auto start = std::chrono::high_resolution_clock::now();

    uint64_t n = graph.num_vertices;
    std::vector<float> pr_curr(n, 1.0f / n);
    std::vector<float> pr_next(n, 0.0f);
    
    // We need out-degrees for the pull formula
    std::vector<uint32_t> out_degrees(n, 0);
    for (uint64_t i = 0; i < graph.num_edges; ++i) {
        out_degrees[graph.column_indices[i]]++;
    }

    float base_rank = (1.0f - damping) / n;

    for (uint32_t iter = 0; iter < max_iterations; ++iter) {
        // Sum PageRank from dangling nodes
        double dangling_sum = 0.0;
        #pragma omp parallel for reduction(+:dangling_sum)
        for (uint32_t i = 0; i < n; ++i) {
            if (out_degrees[i] == 0) {
                dangling_sum += pr_curr[i];
            }
        }
        float dangling_contrib = static_cast<float>(dangling_sum) / n;

        // Pull PageRank
        #pragma omp parallel for
        for (uint32_t v = 0; v < n; ++v) {
            double incoming_sum = 0.0;
            for (uint64_t i = graph.row_offsets[v]; i < graph.row_offsets[v + 1]; ++i) {
                uint32_t u = graph.column_indices[i];
                incoming_sum += pr_curr[u] / out_degrees[u];
            }
            pr_next[v] = base_rank + damping * (static_cast<float>(incoming_sum) + dangling_contrib);
        }

        pr_curr = pr_next;
    }

    auto end = std::chrono::high_resolution_clock::now();
    timings.traversal_ms = std::chrono::duration<double, std::milli>(end - start).count();

    return {pr_curr, timings};
}

bool PageRankValidator::verify(const Graph& graph, 
                              std::span<const float> gpu_results, 
                              std::span<const float> cpu_results,
                              float tolerance) {
    uint32_t mismatches = 0;
    double max_diff = 0.0;

    for (uint32_t i = 0; i < graph.num_vertices; ++i) {
        double diff = std::abs(static_cast<double>(gpu_results[i]) - static_cast<double>(cpu_results[i]));
        max_diff = std::max(max_diff, diff);
        
        if (diff > tolerance) {
            if (mismatches < 10) {
                std::cerr << "Mismatch at vertex " << i << ": GPU=" << gpu_results[i] 
                          << ", CPU=" << cpu_results[i] << " (diff=" << diff << ")" << std::endl;
            }
            mismatches++;
        }
    }

    std::cout << "PageRank Verification Results:" << std::endl;
    std::cout << "  Max Difference: " << max_diff << std::endl;
    
    if (mismatches == 0) {
        std::cout << "SUCCESS: GPU PageRank matches CPU baseline." << std::endl;
        return true;
    } else {
        std::cout << "FAILURE: " << mismatches << " PageRank mismatches found above tolerance " << tolerance << std::endl;
        return false;
    }
}

} // namespace graphs
} // namespace pictor
