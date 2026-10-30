// SPDX-License-Identifier: GPL-2.0-only
// Author: Himanshu Gupta <himanshu3@iisc.ac.in>

#include "cc_validation.hpp"

#include <algorithm>
#include <limits>
#include <numeric>

namespace pictor {
namespace graphs {

namespace {
uint32_t find(std::vector<uint32_t>& parent, uint32_t x)
{
    while (parent[x] != x) {
        parent[x] = parent[parent[x]];
        x         = parent[x];
    }
    return x;
}
} // namespace

CCValidator::Result CCValidator::compute(std::span<const uint64_t> row_offsets,
                                         std::span<const uint32_t> column_indices)
{
    Result result;
    const uint64_t n = row_offsets.empty() ? 0 : (uint64_t)row_offsets.size() - 1;
    if (n == 0)
        return result;

    std::vector<uint32_t> parent(n);
    std::vector<uint32_t> size(n, 1);
    std::iota(parent.begin(), parent.end(), 0u);

    for (uint64_t u = 0; u < n; ++u) {
        for (uint64_t e = row_offsets[u]; e < row_offsets[u + 1]; ++e) {
            const uint32_t v = column_indices[e];
            if (v == u)
                continue;
            uint32_t ru = find(parent, (uint32_t)u);
            uint32_t rv = find(parent, v);
            if (ru == rv)
                continue;
            if (size[ru] < size[rv])
                std::swap(ru, rv);
            parent[rv] = ru;
            size[ru] += size[rv];
        }
    }

    result.labels.resize(n);
    for (uint64_t i = 0; i < n; ++i)
        result.labels[i] = find(parent, (uint32_t)i);

    result.components = 0;
    for (uint64_t i = 0; i < n; ++i)
        result.components += (result.labels[i] == i) ? 1u : 0u;
    return result;
}

bool CCValidator::verify(std::span<const uint32_t> gpu_roots, const Result& cpu)
{
    if (gpu_roots.size() != cpu.labels.size())
        return false;

    uint64_t gpu_components = 0;
    for (uint64_t i = 0; i < gpu_roots.size(); ++i)
        gpu_components += (gpu_roots[i] == i) ? 1u : 0u;
    if (gpu_components != cpu.components)
        return false;

    // Map each CPU component to the GPU root that must contain all of it.
    std::vector<uint32_t> gpu_of_cpu(cpu.labels.size(), std::numeric_limits<uint32_t>::max());
    for (uint64_t i = 0; i < cpu.labels.size(); ++i) {
        const uint32_t c = cpu.labels[i];
        const uint32_t g = gpu_roots[i];
        uint32_t&      m = gpu_of_cpu[c];
        if (m == std::numeric_limits<uint32_t>::max())
            m = g;
        else if (m != g)
            return false;
    }
    return true;
}

} // namespace graphs
} // namespace pictor
