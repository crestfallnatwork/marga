#include "tc.hpp"
#include "csr.hpp"

#include "pictor_core/context/context.hpp"
#include <chrono>

static const uint cMaxMeshDim = 65535;

namespace pictor {
namespace graphs {

TC::TC(Context& ctx) : ctx(ctx)
{
    tc.reset(TCExec::create(ctx).val().release());
}

void TC::init_buffers()
{
    result   = ctx.pool_allocator->allocate(sizeof(uint32_t), 0);
    h_result = ctx.pinned_allocator->allocate(sizeof(uint32_t), 0);
    ctx.fill(result, 0);
    ctx.fsync();
}

TC::~TC()
{
    ctx.deallocate(result);
    ctx.deallocate(h_result);
}

uint64_t TC::run(Graph const& g)
{
    using namespace std::chrono_literals;
    using namespace std::chrono;

    init_buffers();

    // auto starting_points = ctx.pool_allocator->allocate(g.num_vertices * sizeof(uint32_t), 0);
    // auto edges_n = ctx.pool_allocator->allocate(sizeof(uint32_t), 0);
    // edgeSum->record(g.num_vertices, 1, 1,
    //                 {
    //                     .list_offset  = g.d_row_offsets,
    //                     .edge_lists   = g.d_column_indices,
    //                     .new_edge_sum = edges_n,
    //                     .starting_points = starting_points,
    //                     .num_vertices = (uint32_t)g.num_vertices,
    //                 });

    // ctx.fsync();
    // auto t2_degree = high_resolution_clock::now();
    // degree_time = t2_degree - t1_degree;
    //
    volatile uint32_t* res    = reinterpret_cast<uint32_t*>(h_result.mappedPtr);
    uint32_t const cBatchSize = 1 << 24;
    uint64_t total_count      = 0;

    auto tc_time        = 0.s;
    auto t1_tc          = high_resolution_clock::now();

    uint32_t total_tasks = (cBatchSize + tc->taskWgX - 1) / tc->taskWgX;
    uint32_t tasks_x     = total_tasks;
    uint32_t tasks_y     = 1;
    if (total_tasks > cMaxMeshDim) {
        tasks_x = cMaxMeshDim;
        tasks_y = (total_tasks + cMaxMeshDim - 1) / cMaxMeshDim;
    }
    uint32_t edges_left = (g.num_edges % cBatchSize);
    uint32_t total_tasks_left = (edges_left + tc->taskWgX - 1) / tc->taskWgX;
    uint32_t tasks_left_x     = total_tasks_left;
    uint32_t tasks_left_y     = 1;
    if (total_tasks_left > cMaxMeshDim) {
        tasks_left_x = cMaxMeshDim;
        tasks_left_y = (total_tasks_left + cMaxMeshDim - 1) / cMaxMeshDim;
    }
    uint64_t edge_start = 0;
    for (edge_start = 0; edge_start < (g.num_edges / cBatchSize) * cBatchSize;
         edge_start += cBatchSize) {
        tc->record_raw(tasks_x, tasks_y, 1,
                       {
                           .list_offset  = reinterpret_cast<uint64_t*>(g.d_row_offsets.address),
                           .edge_lists   = reinterpret_cast<uint32_t*>(g.d_column_indices.address),
                           .tc           = reinterpret_cast<uint32_t*>(result.address),
                           .num_vertices = g.num_vertices,
                           .edge_start   = edge_start,
                           .edge_num     = cBatchSize,
                       });

        ctx.barrier(result, Access::Write, Access::TransferRead);
        ctx.copy(result, h_result);
        ctx.barrier(result, Access::TransferRead, Access::TransferWrite);
        ctx.fill(result, 0);
        ctx.fsync();

        ctx.invalidate(h_result);
        total_count += *res;
    }
    std::cout << std::format("edge start: {}, total_tasks_left: {}, x: {}, y: {}", edge_start,
                             total_tasks_left, tasks_left_x, tasks_left_y) << std::endl;
    if (total_tasks_left) {
        tc->record_raw(tasks_left_x, tasks_left_y, 1,
                       {
                           .list_offset  = reinterpret_cast<uint64_t*>(g.d_row_offsets.address),
                           .edge_lists   = reinterpret_cast<uint32_t*>(g.d_column_indices.address),
                           .tc           = reinterpret_cast<uint32_t*>(result.address),
                           .num_vertices = g.num_vertices,
                           .edge_start   = edge_start,
                           .edge_num     = edges_left,
                       });
        ctx.barrier(result, Access::Write, Access::TransferRead);
        ctx.copy(result, h_result);
        ctx.barrier(result, Access::TransferRead, Access::TransferWrite);
        ctx.fill(result, 0);
        ctx.fsync();

        ctx.invalidate(h_result);
        total_count += *res;
    }
    auto t2_tc = high_resolution_clock::now();
    tc_time    = t2_tc - t1_tc;
    std::cout << std::format("TC took: {}\n", tc_time);
    std::cout << std::format("TC: {}\n", total_count);
    return *res;
}

} // namespace graphs
} // namespace pictor
