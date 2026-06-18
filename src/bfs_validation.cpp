/*
 * Adapted from GAP Benchmark Suite (Scott Beamer)
 * Adapted by Gemini CLI
 */

#include "bfs_validation.hpp"
#include <atomic>
#include <chrono>
#include <iostream>
#include <vector>
#include <algorithm>
#include <queue>

namespace pictor {
namespace graphs {


std::pair<std::vector<uint32_t>, ValidationTimings> BFSValidator::run(const Graph& graph, uint32_t source) {
    ValidationTimings timings;
    auto start = std::chrono::high_resolution_clock::now();

    uint64_t num_vertices = graph.num_vertices;
    std::vector<uint32_t> distances(num_vertices, 0xFFFFFFFF);
    
    std::vector<uint32_t> current_frontier;
    std::vector<uint32_t> next_frontier;
    
    current_frontier.reserve(num_vertices);
    next_frontier.reserve(num_vertices);

    distances[source] = 0;
    current_frontier.push_back(source);
    
    uint32_t distance = 0;
    
    while(!current_frontier.empty()) {
        for(uint32_t v : current_frontier){
            uint64_t s = graph.row_offsets[v];
            uint64_t e = graph.row_offsets[v+1];
            
            for(uint64_t u = s; u < e; u++){
                uint32_t neighbor = graph.column_indices[u];
                
                if(distances[neighbor] == 0xFFFFFFFF){
                    distances[neighbor] = distance + 1;
                    next_frontier.push_back(neighbor);
                }
            }
        }
        
        distance++;
        current_frontier.swap(next_frontier);
        next_frontier.clear(); 
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    timings.traversal_ms = std::chrono::duration<double, std::milli>(end - start).count();

    return {distances, timings};
}

bool BFSValidator::verify(const Graph& graph, uint32_t source, 
                         std::span<const uint32_t> gpu_distances, 
                         std::span<const uint32_t> cpu_distances) {
    uint32_t mismatches = 0;
    uint32_t gpu_reachable = 0;
    uint32_t cpu_reachable = 0;

    for (uint32_t i = 0; i < graph.num_vertices; ++i) {
        if (gpu_distances[i] != 0xFFFFFFFF) gpu_reachable++;
        if (cpu_distances[i] != 0xFFFFFFFF) cpu_reachable++;

        if (gpu_distances[i] != cpu_distances[i]) {
            if (mismatches < 10) {
                std::cerr << "Mismatch at vertex " << i << ": GPU=" << gpu_distances[i] 
                          << ", CPU=" << cpu_distances[i] << std::endl;
            }
            mismatches++;
        }
    }

    std::cout << "Verification Results:" << std::endl;
    std::cout << "  CPU Reachable: " << cpu_reachable << std::endl;
    std::cout << "  GPU Reachable: " << gpu_reachable << std::endl;
    
    if (mismatches == 0) {
        std::cout << "SUCCESS: GPU distances match CPU baseline." << std::endl;
        return true;
    } else {
        std::cout << "FAILURE: " << mismatches << " distance mismatches found." << std::endl;
        return false;
    }
}

} // namespace graphs
} // namespace pictor
