// SPDX-License-Identifier: GPL-2.0-only
// Author: Himanshu Gupta <himanshu3@iisc.ac.in>

#pragma once

#include <cstdint>
#include <span>

namespace pictor {
namespace graphs {

// Reference triangle count for a symmetrized, sorted CSR: for every edge
// (u, v) with u < v, merge-intersect N(u) and N(v) and count common
// neighbors w > v, so each triangle (u < v < w) is counted exactly once.
class TCValidator {
  public:
    static uint64_t run(std::span<const uint64_t> row_offsets,
                        std::span<const uint32_t> column_indices);

    static bool verify(uint64_t gpu_triangles, uint64_t cpu_triangles);
};

} // namespace graphs
} // namespace pictor
