// SPDX-License-Identifier: GPL-2.0-only
// Author: Himanshu Gupta <himanshu3@iisc.ac.in>

#pragma once

#include "bfs.shader.h"
#include "csr.hpp"
#include <cstdint>
#include <span>

namespace pictor {
struct Context;

namespace graphs {

class BFS {
  public:
    BFS(Context& ctx);
    ~BFS();

    void allocate(uint64_t num_vertices, uint64_t num_edges);
    void run(Graph& g, uint32_t src_vertex);
    std::vector<uint32_t> copy_to_host(uint32_t num_vertices);

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


    void _cleanup_buffers();
};

} // namespace graphs
} // namespace pictor
