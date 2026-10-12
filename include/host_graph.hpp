// SPDX-License-Identifier: GPL-2.0-only
// Author: Himanshu Gupta <himanshu3@iisc.ac.in>

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace pictor {
namespace graphs {

enum class GraphLayout {
    CSR, // Compressed Sparse Row (sorted by source, holds destinations)
    CSC  // Compressed Sparse Column (sorted by destination, holds sources)
};

// Host-side CSR/CSC graph (no device allocations). Backends upload their own
// view of it (Vulkan via Graph, CUDA via the ablation harness).
struct HostGraph {
    uint64_t num_vertices = 0;
    uint64_t num_edges    = 0;
    GraphLayout layout    = GraphLayout::CSR;
    bool symmetrized      = false;
    std::vector<uint64_t> row_offsets;
    std::vector<uint32_t> column_indices;
};

// Import a MatrixMarket file into host CSR/CSC. `symmetrize` forces the
// symmetric view even for non-symmetric files. With `use_cache`, a binary
// cache is read/written next to the file (<file>.csrbin / <file>.sym.csrbin,
// and .cscbin for the CSC layout).
HostGraph import_mtx(const std::string& file, bool symmetrize = false,
                     GraphLayout layout = GraphLayout::CSR, bool use_cache = false);

// Binary cache.
void save_binary(const HostGraph& graph, const std::string& path);
HostGraph load_binary(const std::string& path);

} // namespace graphs
} // namespace pictor
