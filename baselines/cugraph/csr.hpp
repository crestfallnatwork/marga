// SPDX-License-Identifier: GPL-2.0-only
// Author: Himanshu Gupta <himanshu3@iisc.ac.in>

#pragma once
#include <cstdint>
#include <span>
#include <string>

enum class GraphLayout {
    CSR, // Compressed Sparse Row (Sorted by Source, holds Destinations)
    CSC  // Compressed Sparse Column (Sorted by Destination, holds Sources)
};

template <typename vertex_t, typename edge_t>
struct Graph {
    ~Graph();
    Graph();

    bool is_symmetric = false;
    vertex_t num_vertices = 0;
    edge_t num_edges    = 0;
    GraphLayout layout    = GraphLayout::CSR;

    std::span<edge_t> row_offsets;
    std::span<vertex_t> column_indices;

    void save_to_binary(const std::string& prefix) const;
    void load_from_binary(const std::string& prefix);

    static Graph<vertex_t, edge_t> import(std::string const& file, bool symmetrize = false,
                        GraphLayout target_layout = GraphLayout::CSR);

  private:
    void* h_row_offsets;
    void* h_column_indices;
};
