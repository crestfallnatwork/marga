// SPDX-License-Identifier: GPL-2.0-only
// Author: Himanshu Gupta <himanshu3@iisc.ac.in>

#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace pictor {
namespace graphs {

// Exact core-number reference (Batagelj-Zaversnik bin decomposition) and the
// comparisons the kc driver reports. Core numbers are integers, so validation
// is exact.
class KCValidator {
  public:
    struct Result {
        std::vector<uint32_t> core;
        uint32_t              max_core = 0;
        uint64_t              isolated = 0;
    };

    struct Comparison {
        uint64_t checked     = 0;
        uint64_t mismatches  = 0;
        uint64_t first_vertex = 0;
        uint32_t first_gpu   = 0;
        uint32_t first_cpu   = 0;
        uint32_t max_core    = 0;
    };

    // Degrees and core numbers on the symmetric simple view; self-loop entries
    // (u == v) are ignored.
    static Result compute(std::span<const uint64_t> row_offsets,
                          std::span<const uint32_t> column_indices);

    static Comparison compare(std::span<const uint32_t> gpu_core, const Result& cpu);

    // Checks that {v : core[v] >= k} is a k-core: every member has at least k
    // neighbours that are also members. Returns false otherwise; `members` and
    // `min_degree` describe the induced subgraph.
    static bool verify_k_core(std::span<const uint64_t> row_offsets,
                              std::span<const uint32_t> column_indices,
                              std::span<const uint32_t> core, uint32_t k, uint64_t& members,
                              uint64_t& min_degree);
};

} // namespace graphs
} // namespace pictor
