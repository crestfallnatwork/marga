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

    volatile uint32_t* res    = reinterpret_cast<uint32_t*>(h_result.mappedPtr);
    uint32_t const cBatchSize = 1 << 24;
    uint64_t total_count      = 0;

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
    return total_count;
}

} // namespace graphs
} // namespace pictor
