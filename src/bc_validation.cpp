// SPDX-License-Identifier: GPL-2.0-only
// Author: Himanshu Gupta <himanshu3@iisc.ac.in>

#include "bc_validation.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace pictor {
namespace graphs {

BCValidator::BCValidator(std::span<const uint64_t> row_offsets,
                         std::span<const uint32_t> column_indices)
    : row_offsets_(row_offsets), column_indices_(column_indices),
      num_vertices_(row_offsets.empty() ? 0 : row_offsets.size() - 1)
{
    delta_.resize(num_vertices_);
    sigma_.resize(num_vertices_);
    dist_.resize(num_vertices_);
    order_.reserve(num_vertices_);
}

const std::vector<double>& BCValidator::delta(uint32_t source)
{
    if (num_vertices_ == 0 || source >= num_vertices_)
        return delta_;

    std::fill(delta_.begin(), delta_.end(), 0.0);
    std::fill(sigma_.begin(), sigma_.end(), 0.0);
    std::fill(dist_.begin(), dist_.end(), -1);
    order_.clear();

    // Forward: level-synchronous BFS with path counting.
    sigma_[source] = 1.0;
    dist_[source]  = 0;
    order_.push_back(source);
    size_t head = 0;
    while (head < order_.size()) {
        const uint32_t u = order_[head++];
        const double   su = sigma_[u];
        for (uint64_t e = row_offsets_[u]; e < row_offsets_[u + 1]; ++e) {
            const uint32_t w = column_indices_[e];
            if (dist_[w] < 0) {
                dist_[w] = dist_[u] + 1;
                order_.push_back(w);
            }
            if (dist_[w] == dist_[u] + 1)
                sigma_[w] += su;
        }
    }

    // `order_` is in non-decreasing distance because BFS appends as it scans;
    // walking it backwards visits every vertex after all of its successors.
    for (size_t i = order_.size(); i-- > 0;) {
        const uint32_t u = order_[i];
        if (u == source)
            continue;
        double acc = 0.0;
        for (uint64_t e = row_offsets_[u]; e < row_offsets_[u + 1]; ++e) {
            const uint32_t w = column_indices_[e];
            if (dist_[w] == dist_[u] + 1 && sigma_[w] > 0.0)
                acc += sigma_[u] / sigma_[w] * (1.0 + delta_[w]);
        }
        delta_[u] = acc;
    }

    return delta_;
}

BCValidator::Comparison BCValidator::compare(uint32_t source, std::span<const float> gpu,
                                             uint64_t offset, double tolerance)
{
    const std::vector<double>& ref = delta(source);
    Comparison                 cmp;

    const uint64_t n = std::min<uint64_t>(gpu.size(), num_vertices_ > offset ? num_vertices_ - offset : 0);
    for (uint64_t i = 0; i < n; ++i) {
        const double c = ref[offset + i];
        const double g = static_cast<double>(gpu[i]);
        const double d = std::fabs(g - c);
        const double r = d / std::max(1.0, std::fabs(c));
        cmp.checked++;
        cmp.max_rel = std::max(cmp.max_rel, r);
        cmp.max_abs = std::max(cmp.max_abs, d);
        cmp.sum_gpu += g;
        cmp.sum_cpu += c;
        if (r > tolerance)
            cmp.exceeding++;
    }
    return cmp;
}

} // namespace graphs
} // namespace pictor
