// SPDX-License-Identifier: GPL-2.0-only
// Author: Himanshu Gupta <himanshu3@iisc.ac.in>

#include "kc_validation.hpp"

#include <algorithm>
#include <limits>
#include <numeric>

namespace pictor {
namespace graphs {

KCValidator::Result KCValidator::compute(std::span<const uint64_t> row_offsets,
                                         std::span<const uint32_t> column_indices)
{
    Result result;
    const uint64_t n = row_offsets.empty() ? 0 : (uint64_t)row_offsets.size() - 1;
    if (n == 0)
        return result;

    // The bin decomposition keeps the current degree in `core`; when a vertex
    // is processed, that value is its core number.
    std::vector<uint32_t> core(n, 0);
    uint32_t max_degree = 0;
    for (uint64_t v = 0; v < n; ++v) {
        uint32_t d = 0;
        for (uint64_t e = row_offsets[v]; e < row_offsets[v + 1]; ++e)
            d += (column_indices[e] != v) ? 1u : 0u;
        core[v]    = d;
        max_degree = std::max(max_degree, d);
        result.isolated += (d == 0) ? 1u : 0u;
    }

    // bin[d] tracks the boundary between the vertices of degree < d and >= d.
    std::vector<uint64_t> bin(max_degree + 2, 0);
    for (uint64_t v = 0; v < n; ++v)
        bin[core[v]]++;
    uint64_t start = 0;
    for (uint32_t d = 0; d <= max_degree; ++d) {
        const uint64_t count = bin[d];
        bin[d]               = start;
        start += count;
    }
    bin[max_degree + 1] = start;

    std::vector<uint32_t> vert(n);
    std::vector<uint32_t> pos(n);
    {
        std::vector<uint64_t> next(bin.begin(), bin.end());
        for (uint64_t v = 0; v < n; ++v) {
            pos[v]       = (uint32_t)next[core[v]];
            vert[pos[v]] = (uint32_t)v;
            next[core[v]]++;
        }
    }

    // Every edge is examined once, from the endpoint processed first; the
    // processed prefix of `vert` is final and is never touched again.
    std::vector<uint8_t> processed(n, 0);
    for (uint64_t i = 0; i < n; ++i) {
        const uint32_t v = vert[i];
        processed[v]     = 1;
        for (uint64_t e = row_offsets[v]; e < row_offsets[v + 1]; ++e) {
            const uint32_t u = column_indices[e];
            if (u == v || processed[u] != 0)
                continue;
            if (core[u] <= core[v])
                continue;

            const uint32_t du = core[u];
            const uint64_t pu = pos[u];
            const uint64_t pw = bin[du];
            const uint32_t w  = vert[pw];

            vert[pu] = w;
            pos[w]   = (uint32_t)pu;
            vert[pw] = u;
            pos[u]   = (uint32_t)pw;

            bin[du] += 1;
            core[u] = du - 1;
        }
    }

    for (uint64_t v = 0; v < n; ++v)
        result.max_core = std::max(result.max_core, core[v]);
    result.core = std::move(core);
    return result;
}

KCValidator::Comparison KCValidator::compare(std::span<const uint32_t> gpu_core, const Result& cpu)
{
    Comparison cmp;
    const uint64_t n = std::min<uint64_t>(gpu_core.size(), cpu.core.size());
    cmp.checked      = n;
    cmp.max_core     = cpu.max_core;
    for (uint64_t v = 0; v < n; ++v) {
        if (gpu_core[v] == cpu.core[v])
            continue;
        if (cmp.mismatches == 0) {
            cmp.first_vertex = v;
            cmp.first_gpu    = gpu_core[v];
            cmp.first_cpu    = cpu.core[v];
        }
        ++cmp.mismatches;
    }
    if (gpu_core.size() != cpu.core.size())
        ++cmp.mismatches;
    return cmp;
}

bool KCValidator::verify_k_core(std::span<const uint64_t> row_offsets,
                                std::span<const uint32_t> column_indices,
                                std::span<const uint32_t> core, uint32_t k, uint64_t& members,
                                uint64_t& min_degree)
{
    const uint64_t n = core.size();
    members          = 0;
    min_degree       = std::numeric_limits<uint64_t>::max();

    for (uint64_t v = 0; v < n; ++v) {
        if (core[v] < k)
            continue;
        ++members;
        uint64_t degree = 0;
        for (uint64_t e = row_offsets[v]; e < row_offsets[v + 1]; ++e) {
            const uint32_t u = column_indices[e];
            if (u != v && core[u] >= k)
                ++degree;
        }
        min_degree = std::min(min_degree, degree);
    }

    if (members == 0) {
        min_degree = 0;
        return true;
    }
    return min_degree >= k;
}

} // namespace graphs
} // namespace pictor
