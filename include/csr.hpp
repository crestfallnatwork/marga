// SPDX-License-Identifier: GPL-2.0-only
// Author: Himanshu Gupta <himanshu3@iisc.ac.in>

#pragma once

#include "host_graph.hpp"
#include "pictor_core/memory/allocation_type.hpp"

#include <cstdint>
#include <span>
#include <string>

namespace pictor {
struct Context;

namespace graphs {

struct Graph {
    ~Graph();
    Graph(Context* ctx);

    AllocationInfo d_row_offsets;
    AllocationInfo d_column_indices;

    uint64_t num_vertices = 0;
    uint64_t num_edges    = 0;
    GraphLayout layout    = GraphLayout::CSR;
    bool symmetrized      = false;

    std::span<uint64_t> row_offsets;
    std::span<uint32_t> column_indices;

    void save_to_binary(const std::string& path) const;
    void load_from_binary(const std::string& path);

    static Graph import(Context& ctx, std::string const& file, bool symmetrize = false,
                        GraphLayout target_layout = GraphLayout::CSR);

  private:
    // Copy host CSR/CSC into pinned buffers and upload to the device.
    void upload_host_graph(const HostGraph& graph);

    AllocationInfo h_row_offsets;
    AllocationInfo h_column_indices;
    Context* ctx;
};
} // namespace graphs
} // namespace pictor
