// SPDX-License-Identifier: GPL-2.0-only
// Author: Himanshu Gupta <himanshu3@iisc.ac.in>

#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace pictor {
namespace graphs {

// Reference single-source Brandes over the same CSR the GPU sees: double
// precision, raw dependency values (no normalization), so a GPU row can be
// compared directly.
class BCValidator {
  public:
    struct Comparison {
        uint64_t checked   = 0;
        uint64_t exceeding = 0;  // |gpu - cpu| > tolerance * max(1, |cpu|)
        double   max_rel   = 0.0;
        double   max_abs   = 0.0;
        double   sum_gpu   = 0.0;
        double   sum_cpu   = 0.0;
    };

    BCValidator(std::span<const uint64_t> row_offsets, std::span<const uint32_t> column_indices);

    // Raw dependencies of every vertex for `source` (length = vertices).
    const std::vector<double>& delta(uint32_t source);

    uint64_t vertices() const { return num_vertices_; }
    uint64_t edges() const { return column_indices_.size(); }

    // Compares `count` GPU values for `source` starting at vertex `offset`
    // against the reference row.
    Comparison compare(uint32_t source, std::span<const float> gpu, uint64_t offset,
                       double tolerance);

  private:
    std::span<const uint64_t> row_offsets_;
    std::span<const uint32_t> column_indices_;
    uint64_t                  num_vertices_ = 0;

    std::vector<double> delta_;
    std::vector<double> sigma_;
    std::vector<int64_t> dist_;
    std::vector<uint32_t> order_;
};

} // namespace graphs
} // namespace pictor
