// SPDX-License-Identifier: GPL-2.0-only
// Author: Himanshu Gupta <himanshu3@iisc.ac.in>

#include "tc_validation.hpp"

#include <thread>
#include <vector>

namespace pictor {
namespace graphs {

uint64_t TCValidator::run(std::span<const uint64_t> row_offsets,
                          std::span<const uint32_t> column_indices)
{
    const uint64_t n = row_offsets.empty() ? 0 : (uint64_t)row_offsets.size() - 1;
    if (n == 0)
        return 0;

    unsigned workers = std::thread::hardware_concurrency();
    if (workers == 0)
        workers = 1;
    if ((uint64_t)workers > n)
        workers = (unsigned)n;

    // One contiguous vertex range per worker; the reduction is a handful of
    // adds instead of a per-vertex array.
    std::vector<uint64_t>    chunk_sums(workers, 0);
    std::vector<std::thread> threads;
    for (unsigned w = 0; w < workers; ++w) {
        threads.emplace_back([&, w]() {
            const uint64_t a     = n * w / workers;
            const uint64_t b     = n * (w + 1) / workers;
            uint64_t       local = 0;
            for (uint64_t u = a; u < b; ++u) {
                const uint64_t su = row_offsets[u];
                const uint64_t eu = row_offsets[u + 1];
                for (uint64_t i = su; i < eu; ++i) {
                    const uint32_t v = column_indices[i];
                    if ((uint64_t)v <= u)
                        continue;
                    const uint64_t sv = row_offsets[v];
                    const uint64_t ev = row_offsets[v + 1];
                    uint64_t       ia = su, ib = sv;
                    while (ia < eu && ib < ev) {
                        const uint32_t wa = column_indices[ia];
                        const uint32_t wb = column_indices[ib];
                        if (wa == wb) {
                            if ((uint64_t)wa > (uint64_t)v)
                                ++local; // u < v < w: count once
                            ++ia;
                            ++ib;
                        } else if (wa < wb) {
                            ++ia;
                        } else {
                            ++ib;
                        }
                    }
                }
            }
            chunk_sums[w] = local;
        });
    }
    for (auto& t : threads)
        t.join();

    uint64_t total = 0;
    for (uint64_t s : chunk_sums)
        total += s;
    return total;
}

bool TCValidator::verify(uint64_t gpu_triangles, uint64_t cpu_triangles)
{
    return gpu_triangles == cpu_triangles;
}

} // namespace graphs
} // namespace pictor
