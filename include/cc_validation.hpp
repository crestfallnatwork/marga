// SPDX-License-Identifier: GPL-2.0-only
// Author: Himanshu Gupta <himanshu3@iisc.ac.in>

#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace pictor {
namespace graphs {

// Reference connected components by union-find over the same CSR the GPU sees
// (weak components for a directed view). Labels are component representatives.
class CCValidator {
  public:
    struct Result {
        uint64_t              components = 0;
        std::vector<uint32_t> labels;
    };

    static Result compute(std::span<const uint64_t> row_offsets,
                          std::span<const uint32_t> column_indices);

    // Exact partition check: every vertex whose CPU label matches must also
    // share one GPU root, and the component counts must agree.
    static bool verify(std::span<const uint32_t> gpu_roots, const Result& cpu);
};

} // namespace graphs
} // namespace pictor
