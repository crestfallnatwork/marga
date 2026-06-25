// SPDX-License-Identifier: GPL-2.0-only
// Author: Himanshu Gupta <himanshu3@iisc.ac.in>

// Shader for dynamically calculating out-degrees from a CSC graph

struct Params {
    uint64_t row_indices; // Source vertices for each edge (CSC format)
    uint64_t out_degrees; // Array of size num_vertices to store output
    uint64_t num_edges;   // Total number of edges in the graph
};

[[vk::push_constant]]
uniform Params p;

[shader("compute")]
[numthreads(256, 1, 1)]
void calc_out_degrees(uint32_t edge_idx : SV_DispatchThreadID) {
    if (edge_idx >= p.num_edges) return;

    // In a CSC graph, the row_indices array contains the original source vertex
    // for every edge in the graph. We read the source vertex for this specific edge.
    uint32_t* row_indices = (uint32_t*)p.row_indices;
    uint32_t source_vertex = row_indices[edge_idx];

    // Atomically increment the out-degree counter for this source vertex.
    // This perfectly counts how many times a vertex appears as a source across the entire graph.
    uint32_t* out_degrees = (uint32_t*)p.out_degrees;
    InterlockedAdd(out_degrees[source_vertex], 1);
}
