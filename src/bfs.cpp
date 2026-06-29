// SPDX-License-Identifier: GPL-2.0-only
// Author: Himanshu Gupta <himanshu3@iisc.ac.in>

#include "bfs.hpp"
#include "pictor_core/context/context.hpp"
#include "pictor_core/execution/synchronization.hpp"

// Generated headers from Slang
#include <chrono>

namespace pictor {
namespace graphs {

BFS::BFS(Context& ctx) : _ctx(ctx)
{
    bfs.reset(bfs_shader::BFSExecutor::create(ctx).val().release());
    cbfs.reset(bfs_shader::ComputeComputeExecutor::create(ctx).val().release());
    dispatch.reset(bfs_shader::DispatchComputeExecutor::create(ctx).val().release());
}

BFS::~BFS()
{
    _cleanup_buffers();
}

void BFS::_allocate_buffers(uint64_t num_vertices, uint64_t num_edges)
{
    _cleanup_buffers();

    _frontier_a = _ctx.pool_allocator->allocate(num_vertices * sizeof(uint32_t), alignof(uint32_t));
    _frontier_b = _ctx.pool_allocator->allocate(num_vertices * sizeof(uint32_t), alignof(uint32_t));
    _frontier_size_a = _ctx.pool_allocator->allocate(sizeof(uint32_t), alignof(uint32_t));
    _frontier_size_b = _ctx.pool_allocator->allocate(sizeof(uint32_t), alignof(uint32_t));

    uint64_t visited_size = (num_vertices + 31) / 32 * sizeof(uint32_t);
    _visited              = _ctx.pool_allocator->allocate(visited_size, alignof(uint32_t));
    _distances = _ctx.pool_allocator->allocate(num_vertices * sizeof(uint32_t), alignof(uint32_t));
    _h_distances =
        _ctx.pinned_allocator->allocate(num_vertices * sizeof(uint32_t), alignof(uint32_t));
    _indirect_buffer = _ctx.pool_allocator->allocate(4 * sizeof(uint32_t), alignof(uint32_t));
    _staging_size    = _ctx.pinned_allocator->allocate(sizeof(uint32_t), alignof(uint32_t));
}

void BFS::_cleanup_buffers()
{
    if (_frontier_a.allocation)
        _ctx.deallocate(_frontier_a);
    if (_frontier_b.allocation)
        _ctx.deallocate(_frontier_b);
    if (_frontier_size_a.allocation)
        _ctx.deallocate(_frontier_size_a);
    if (_frontier_size_b.allocation)
        _ctx.deallocate(_frontier_size_b);
    if (_visited.allocation)
        _ctx.deallocate(_visited);
    if (_distances.allocation)
        _ctx.deallocate(_distances);
    if (_h_distances.allocation)
        _ctx.deallocate(_h_distances);
    if (_indirect_buffer.allocation)
        _ctx.deallocate(_indirect_buffer);
    if (_staging_size.allocation)
        _ctx.deallocate(_staging_size);
}

std::pair<std::span<uint32_t>, BFSTimings> BFS::run(Graph& g, uint32_t src_vertex)
{
    auto t_full_start = std::chrono::high_resolution_clock::now();
    BFSTimings t;

    auto t_alloc_start = std::chrono::high_resolution_clock::now();
    if (_h_distances.size != g.num_vertices * sizeof(uint32_t)) {
        _allocate_buffers(g.num_vertices, g.num_edges);
    }
    auto t_alloc_end = std::chrono::high_resolution_clock::now();
    t.alloc_ms = std::chrono::duration<double, std::milli>(t_alloc_end - t_alloc_start).count();

    auto t_init_start = std::chrono::high_resolution_clock::now();
    _ctx.fill(_distances, 0xFFFFFFFF);
    _ctx.fill(_visited, 0);
    _ctx.fill(_frontier_size_a, 1);
    _ctx.fill(_frontier_size_b, 0);
    _ctx.fill(_indirect_buffer, 1);

    uint32_t one  = 1;
    uint32_t zero = 0;
    _ctx.update(_frontier_a, 0, sizeof(uint32_t), &src_vertex);
    _ctx.update(_distances, src_vertex * sizeof(uint32_t), sizeof(uint32_t), &zero);

    uint32_t word_idx = src_vertex / 32;
    uint32_t mask     = 1 << (src_vertex % 32);
    _ctx.update(_visited, word_idx * sizeof(uint32_t), sizeof(uint32_t), &mask);

    _ctx.fsync();

    auto t_init_end = std::chrono::high_resolution_clock::now();
    t.init_ms       = std::chrono::duration<double, std::milli>(t_init_end - t_init_start).count();

    uint32_t current_level                        = 0;
    bool finished                                 = false;
    uint32_t next_size                            = 1;
    std::array<AllocationInfo*, 2> frontiers      = {&_frontier_a, &_frontier_b};
    std::array<AllocationInfo*, 2> frontier_sizes = {&_frontier_size_a, &_frontier_size_b};

    auto t_traversal_start = std::chrono::high_resolution_clock::now();

    uint32_t batch_size = 10;
    int current         = 0;
    int next            = 1;
    while (next_size != 0) {
        for (int i = 0; i < batch_size; i++) {
            current = current_level % 2;
            next    = (current_level + 1) % 2;

            dispatch->record(
                1, 1, 1,
                {
                    .frontier_size = reinterpret_cast<uint32_t*>(frontier_sizes[current]->address),
                    .grid          = reinterpret_cast<uint32_t*>(_indirect_buffer.address),
                });

            _ctx.fill(*frontier_sizes[next], 0);
            _ctx.barrier(*frontier_sizes[next], Access::TransferWrite, Access::Write);
            _ctx.barrier(_indirect_buffer, Access::Write, Access::IndirectRead);

            BFSIn bfsIn = {
                .current_level    = current_level,
                .row_offsets      = reinterpret_cast<uint64_t*>(g.d_row_offsets.address),
                .column_indices   = reinterpret_cast<uint32_t*>(g.d_column_indices.address),
                .visited          = reinterpret_cast<uint32_t*>(_visited.address),
                .distances        = reinterpret_cast<uint32_t*>(_distances.address),
                .current_frontier = reinterpret_cast<uint32_t*>(frontiers[current]->address),
                .next_frontier    = reinterpret_cast<uint32_t*>(frontiers[next]->address),
                .current_frontier_size =
                    reinterpret_cast<uint32_t*>(frontier_sizes[current]->address),
                .next_frontier_size = reinterpret_cast<uint32_t*>(frontier_sizes[next]->address),
                .num_vertices       = g.num_vertices,
            };
            bfs->record_indirect(_indirect_buffer, bfsIn);
            // bfs->record(next_size, 1, 1, bfsIn);
            // cbfs->record(next_size, 1, 1, bfsIn); /* vertex variant */
            _ctx.barrier({ResourceBarrier{frontiers[next], Access::Write, Access::Read},
                          ResourceBarrier{&_visited, Access::Write, Access::Read},
                          ResourceBarrier{frontier_sizes[next], Access::Write, Access::Read}});
            current_level++;
        }

        _ctx.copy(*frontier_sizes[next], _staging_size);
        _ctx.fsync();
        _ctx.invalidate(_staging_size);
        next_size = *reinterpret_cast<uint32_t*>(_staging_size.mappedPtr);
        // std::cout << std::format("Level: {}, Frontier size: {}\n", current_level,
        // next_size);
    }

    auto t_traversal_end = std::chrono::high_resolution_clock::now();
    t.traversal_ms =
        std::chrono::duration<double, std::milli>(t_traversal_end - t_traversal_start).count();

    auto t_readback_st = std::chrono::high_resolution_clock::now();

    _ctx.copy(_distances, _h_distances);
    _ctx.fsync();

    auto t_readback_end = std::chrono::high_resolution_clock::now();
    t.pci_readback_ms =
        std::chrono::duration<double, std::milli>(t_readback_end - t_readback_st).count();
    auto t_full_end = std::chrono::high_resolution_clock::now();
    t.total_ms      = std::chrono::duration<double, std::milli>(t_full_end - t_full_start).count();

    return {std::span<uint32_t>(static_cast<uint32_t*>(_h_distances.mappedPtr), g.num_vertices), t};
}

} // namespace graphs
} // namespace pictor
