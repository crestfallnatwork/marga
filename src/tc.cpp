#include "tc.hpp"
#include "csr.hpp"

#include "pictor_core/context/context.hpp"
#include "pictor_core/execution/gpu_timer.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>

static const uint32_t cMaxMeshDim = 65535;

namespace pictor {
namespace graphs {

TC::TC(Context& ctx, uint32_t cull, uint64_t batch_edges)
    : ctx(ctx), cull(cull), batch_edges(batch_edges)
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

    const auto run_t0 = high_resolution_clock::now();

    // Optional instrumentation (ablation attribution; off by default).
    const bool     profiling  = std::getenv("MARGA_TC_TIMESTAMPS") != nullptr;
    const char*    tn_env     = std::getenv("MARGA_TC_TASK_NOOP");
    const uint32_t task_noop  = tn_env ? static_cast<uint32_t>(std::strtoul(tn_env, nullptr, 10)) : 0u;
    const uint32_t mesh_noop  = (task_noop || std::getenv("MARGA_TC_MESH_NOOP")) ? 1u : 0u;
    const uint32_t no_payload = std::getenv("MARGA_TC_NO_PAYLOAD") ? 1u : 0u;
    const bool     count_mesh = std::getenv("MARGA_TC_COUNT_MESH") != nullptr;

    static uint32_t s_run_index = 0;
    ++s_run_index;

    init_buffers();

    AllocationInfo mesh_count_dev{};
    AllocationInfo mesh_count_host{};
    if (count_mesh) {
        mesh_count_dev  = ctx.pool_allocator->allocate(sizeof(uint64_t), 0);
        mesh_count_host = ctx.pinned_allocator->allocate(sizeof(uint64_t), 0);
        ctx.fill(mesh_count_dev, 0);
        ctx.fsync();
    }

    volatile uint32_t* res    = reinterpret_cast<uint32_t*>(h_result.mappedPtr);
    const uint64_t cBatchSize = batch_edges;
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

    const uint64_t num_batches = (g.num_edges + cBatchSize - 1) / cBatchSize;

    GpuTimer timer;
    if (profiling) {
        auto timer_res = GpuTimer::create(ctx, static_cast<uint32_t>(3 * num_batches));
        if (!timer_res.ok()) {
            std::fprintf(stderr, "tc: failed to create the GPU timer\n");
            std::exit(1);
        }
        timer = std::move(timer_res.val());
        timer.reset();
    }

    double   host_record_us = 0.0;
    double   host_wait_us   = 0.0;
    uint64_t batch_index    = 0;
    uint64_t task_groups    = 0;

    for (uint64_t edge_start = 0; edge_start < g.num_edges; edge_start += cBatchSize) {
        const uint64_t edge_num = std::min<uint64_t>(cBatchSize, g.num_edges - edge_start);
        const uint32_t tx       = (edge_num == cBatchSize) ? tasks_x : tasks_left_x;
        const uint32_t ty       = (edge_num == cBatchSize) ? tasks_y : tasks_left_y;
        task_groups += static_cast<uint64_t>(tx) * ty;

        const auto t_rec0 = high_resolution_clock::now();
        {
            auto lap = timer.lap("draw");
            tc->record_raw(tx, ty, 1,
                           {
                               .list_offset  = reinterpret_cast<uint64_t*>(g.d_row_offsets.address),
                               .edge_lists   = reinterpret_cast<uint32_t*>(g.d_column_indices.address),
                               .tc           = reinterpret_cast<uint32_t*>(result.address),
                               .num_vertices = g.num_vertices,
                               .edge_start   = edge_start,
                               .edge_num     = edge_num,
                               .mesh_noop    = mesh_noop,
                               .no_payload   = no_payload,
                               .task_noop    = task_noop,
                               .count_mesh   = count_mesh ? 1u : 0u,
                               .mesh_count   = reinterpret_cast<uint64_t*>(mesh_count_dev.address),
                               .cull         = cull,
                           });
        }

        {
            auto lap = timer.lap("copy");
            ctx.barrier(result, Access::Write, Access::TransferRead);
            ctx.copy(result, h_result);
        }

        {
            auto lap = timer.lap("fill");
            ctx.barrier(result, Access::TransferRead, Access::TransferWrite);
            ctx.fill(result, 0);
        }
        const auto t_rec1 = high_resolution_clock::now();

        ctx.fsync();
        const auto t_wait1 = high_resolution_clock::now();

        ctx.invalidate(h_result);
        total_count += *res;

        if (profiling) {
            host_record_us += duration<double, std::micro>(t_rec1 - t_rec0).count();
            host_wait_us += duration<double, std::micro>(t_wait1 - t_rec1).count();
        }
        ++batch_index;
    }

    if (count_mesh) {
        ctx.copy(mesh_count_dev, mesh_count_host);
        ctx.fsync();
        ctx.invalidate(mesh_count_host);
        const uint64_t groups = *reinterpret_cast<uint64_t*>(mesh_count_host.mappedPtr);
        std::printf("TSMESH run=%u groups=%llu task_groups=%llu groups_per_task=%.3f\n",
                    s_run_index, static_cast<unsigned long long>(groups),
                    static_cast<unsigned long long>(task_groups),
                    task_groups ? static_cast<double>(groups) / static_cast<double>(task_groups)
                                : 0.0);
        ctx.deallocate(mesh_count_dev);
        ctx.deallocate(mesh_count_host);
    }

    if (profiling && timer.enabled()) {
        auto laps_res = timer.read();
        if (!laps_res.ok() || laps_res.val().size() < 3 * batch_index) {
            std::fprintf(stderr, "tc: GPU timer results unavailable\n");
            std::exit(1);
        }
        const auto laps    = laps_res.val();
        double     draw_us = 0.0;
        double     copy_us = 0.0;
        double     fill_us = 0.0;
        for (uint64_t b = 0; b < batch_index; ++b) {
            const double d_draw = laps[3 * b + 0].ms * 1000.0;
            const double d_copy = laps[3 * b + 1].ms * 1000.0;
            const double d_fill = laps[3 * b + 2].ms * 1000.0;
            draw_us += d_draw;
            copy_us += d_copy;
            fill_us += d_fill;
            std::printf("TSBATCH run=%u batch=%llu draw_us=%.1f copy_us=%.1f fill_us=%.1f\n",
                        s_run_index, static_cast<unsigned long long>(b), d_draw, d_copy, d_fill);
        }
        const auto   run_t1 = high_resolution_clock::now();
        const double run_us = duration<double, std::micro>(run_t1 - run_t0).count();
        std::printf("TSPHASE run=%u batches=%llu draw_us=%.1f copy_us=%.1f fill_us=%.1f "
                    "gpu_us=%.1f record_us=%.1f wait_us=%.1f run_us=%.1f mesh_noop=%u "
                    "no_payload=%u task_noop=%u\n",
                    s_run_index, static_cast<unsigned long long>(batch_index), draw_us, copy_us,
                    fill_us, draw_us + copy_us + fill_us, host_record_us, host_wait_us, run_us,
                    mesh_noop, no_payload, task_noop);
    }
    return total_count;
}

} // namespace graphs
} // namespace pictor
