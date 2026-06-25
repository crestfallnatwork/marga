// SPDX-License-Identifier: GPL-2.0-only
// Author: Himanshu Gupta <himanshu3@iisc.ac.in>

#pragma once

#include "csr.hpp"
#include "bfs.shader.h"
#include <cstdint>
#include <span>

namespace pictor {
struct Context;

namespace graphs {


struct BFSTimings {
    double alloc_ms = 0;
    double init_ms = 0;
    double traversal_ms = 0;
    double inner_sync_ms = 0;
    double pci_readback_ms = 0;
    double total_ms = 0;
};

class BFS {
public:
    BFS(Context& ctx);
    ~BFS();

    // Returns a span to host-pinned distances and detailed timing information.
    std::pair<std::span<uint32_t>, BFSTimings> run(Graph& g, uint32_t src_vertex);

private:
    Context& _ctx;

    std::unique_ptr<bfs_shader::BFSExecutor> bfs;
    std::unique_ptr<bfs_shader::ComputeComputeExecutor> cbfs;
    using BFSIn = bfs_shader::BFSIn;

    std::unique_ptr<bfs_shader::DispatchComputeExecutor> dispatch;
    using dispatchIn = bfs_shader::DispatchIn;
    
    AllocationInfo _frontier_a;
    AllocationInfo _frontier_b;
    AllocationInfo _frontier_size_a;
    AllocationInfo _frontier_size_b;
    
    AllocationInfo _visited;
    AllocationInfo _distances;
    AllocationInfo _h_distances;
    AllocationInfo _indirect_buffer;
    AllocationInfo _staging_size;

    void _allocate_buffers(uint64_t num_vertices, uint64_t num_edges);
    void _cleanup_buffers();
};

} // namespace graphs
} // namespace pictor
