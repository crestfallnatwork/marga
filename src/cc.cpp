// SPDX-License-Identifier: GPL-2.0-only
// Author: Himanshu Gupta <himanshu3@iisc.ac.in>

#include "cc.hpp"
#include "csr.hpp"
#include "helper.hpp"

#include "pictor_core/context/context.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <format>

static const uint32_t cMaxMeshDim = 65535;
static const uint64_t cEdgeBatch  = 1ull << 24;

namespace pictor {
namespace graphs {

namespace {
// Must match cEdgesPerLane * WAVE_SIZE in cc.slang.
constexpr uint64_t cTaskEdges = 256;

void split_grid(uint32_t groups, uint32_t& x, uint32_t& y)
{
    if (groups > cMaxMeshDim) {
        x = cMaxMeshDim;
        y = (groups + cMaxMeshDim - 1) / cMaxMeshDim;
    } else {
        x = groups;
        y = 1;
    }
}
} // namespace

CC::CC(Context& ctx) : ctx(ctx)
{
    exec.reset(CCExec::create(ctx).val().release());
    prepare_exec.reset(PrepareExec::create(ctx).val().release());
}

CC::~CC()
{
    release();
    ctx.deallocate(h_counters);
    ctx.deallocate(h_parent);
}

void CC::release()
{
    ctx.deallocate(d_parent);
    ctx.deallocate(d_active[0]);
    ctx.deallocate(d_active[1]);
    ctx.deallocate(d_counters);
}

void CC::prepare(Graph const& g)
{
    num_vertices_ = g.num_vertices;
    num_edges_    = g.num_edges;

    // parent + the first active pair list (two u32 per entry). The first list
    // is capped below the edge count so the second list can coexist; the
    // second is sized by the live count the first round measures. A round
    // whose live count does not fit its destination list is not stored, and
    // the host rescans the CSR into the same buffer instead.
    d_parent    = ctx.pool_allocator->allocate(num_vertices_ * sizeof(uint32_t), 0);
    d_counters  = ctx.pool_allocator->allocate(2 * sizeof(uint32_t), 0);
    CheckAllocation(d_parent, "cc: parent");
    CheckAllocation(d_counters, "cc: counters");

    const uint64_t avail = static_cast<uint64_t>(ctx.get_device_free_memory() * 0.95);
    uint64_t cap = std::min<uint64_t>(num_edges_, (avail * 55 / 100) / (2 * sizeof(uint32_t)));
    if (cap == 0)
        cap = 1;
    for (;;) {
        d_active[0] = ctx.pool_allocator->allocate(cap * 2 * sizeof(uint32_t), 0);
        if (d_active[0].res == vk::Result::eSuccess)
            break;
        if (cap <= (1ull << 22))
            CheckAllocation(d_active[0], "cc: active list 0");
        cap /= 2;
    }

    if (!h_counters.mappedPtr)
        h_counters = ctx.pinned_allocator->allocate(2 * sizeof(uint32_t), 0);

    ctx.fill(d_counters, 0);

    // Parent identity.
    const uint64_t groups = (num_vertices_ + 255) / 256;
    uint32_t       gx     = static_cast<uint32_t>(std::min<uint64_t>(groups, cMaxMeshDim));
    uint32_t       gy     = static_cast<uint32_t>((groups + gx - 1) / gx);
    const uint64_t stride = static_cast<uint64_t>(gx) * 256;

    prepare_exec->record(static_cast<uint32_t>(gx) * 256, gy, 1,
                         {
                             .parent       = reinterpret_cast<uint32_t*>(d_parent.address),
                             .num_vertices = num_vertices_,
                             .stride       = stride,
                         });
    ctx.barrier(d_parent, Access::Write, Access::Read);
    ctx.fsync();
}

uint32_t CC::run(Graph const& g, bool symmetric_view)
{
    using namespace std::chrono;

    release();
    rounds_ = 0;

    const auto init_t0 = high_resolution_clock::now();
    prepare(g);
    init_ms_ = duration<double, std::milli>(high_resolution_clock::now() - init_t0).count();

    const auto loop_t0 = high_resolution_clock::now();

    // `scan` means the next round reads the CSR (no list); otherwise it reads
    // the pair list left by the previous round.
    bool            scan = true;
    AllocationInfo* in   = nullptr;
    AllocationInfo* out  = &d_active[0];
    uint64_t        active_count = num_edges_;

    for (;;) {
        const auto round_t0 = high_resolution_clock::now();
        const uint64_t items = scan ? num_edges_ : active_count;
        const uint64_t capacity = out->size / (2 * sizeof(uint32_t));

        ctx.fill(d_counters, 0);
        ctx.barrier(d_counters, Access::TransferWrite, Access::Read);

        for (uint64_t offset = 0; offset < items; offset += cEdgeBatch) {
            const uint64_t n      = std::min<uint64_t>(cEdgeBatch, items - offset);
            const uint32_t groups = static_cast<uint32_t>((n + cTaskEdges - 1) / cTaskEdges);
            uint32_t       gx, gy;
            split_grid(groups, gx, gy);

            exec->record_raw(gx, gy, 1,
                             {
                                 .row_offsets = reinterpret_cast<uint64_t*>(g.d_row_offsets.address),
                                 .edge_lists  = reinterpret_cast<uint32_t*>(g.d_column_indices.address),
                                 .parent      = reinterpret_cast<uint32_t*>(d_parent.address),
                                 .active      = in ? reinterpret_cast<uint32_t*>(in->address) : nullptr,
                                 .next_active = reinterpret_cast<uint32_t*>(out->address),
                                 .counters    = reinterpret_cast<uint32_t*>(d_counters.address),
                                 .num_vertices = num_vertices_,
                                 .edge_start  = offset,
                                 .edge_num    = n,
                                 .use_active  = scan ? 0u : 1u,
                                 .cull_upper  = symmetric_view ? 1u : 0u,
                                 .list_capacity = capacity,
                             });
        }
        // Make this round's parent and active-list writes visible to the
        // next round's task stage.
        ctx.barrier(d_parent, Access::Write, Access::Read);
        ctx.barrier(*out, Access::Write, Access::Read);
        ctx.barrier(d_counters, Access::Write, Access::TransferRead);

        ctx.copy(d_counters, h_counters);
        ctx.fsync();
        ctx.invalidate(h_counters);
        ctx.barrier(d_counters, Access::TransferRead, Access::TransferWrite);

        const uint32_t* counters = reinterpret_cast<const uint32_t*>(h_counters.mappedPtr);
        const uint32_t  produced = counters[1];
        ++rounds_;

        if (std::getenv("MARGA_CC_TRACE"))
            std::fprintf(stderr, "cc round %u: %s in=%llu produced=%u cap=%llu wall=%.3f ms\n",
                         rounds_, scan ? "scan" : "list",
                         static_cast<unsigned long long>(items), produced,
                         static_cast<unsigned long long>(capacity),
                         std::chrono::duration<double, std::milli>(
                             std::chrono::high_resolution_clock::now() - round_t0)
                             .count());

        // No live pairs remain: every edge's endpoints share a root.
        if (produced == 0)
            break;

        // The list just written is usable only if it fits; otherwise fall back
        // to a full CSR scan, reusing the same buffer.
        if (produced > capacity) {
            scan         = true;
            in           = nullptr;
            active_count = num_edges_;
            continue;
        }

        AllocationInfo* nxt  = (out == &d_active[0]) ? &d_active[1] : &d_active[0];
        const uint64_t  need = static_cast<uint64_t>(produced) * 2 * sizeof(uint32_t);
        if (!nxt->buffer || nxt->size < need) {
            if (nxt->buffer)
                ctx.deallocate(*nxt);
            AllocationInfo next = ctx.pool_allocator->allocate(need, 0);
            if (next.res != vk::Result::eSuccess) {
                // No room for a second list: rescan the CSR into `out`.
                scan         = true;
                in           = nullptr;
                active_count = num_edges_;
                continue;
            }
            *nxt = next;
        }

        in           = out;
        out          = nxt;
        scan         = false;
        active_count = produced;

        if (rounds_ > 4096) {
            std::fprintf(stderr, "cc: warning: round cap reached\n");
            break;
        }
    }

    ctx.fsync();
    loop_ms_ = duration<double, std::milli>(high_resolution_clock::now() - loop_t0).count();

    return rounds_;
}

std::vector<uint32_t> CC::resolve_roots()
{
    const uint64_t n = num_vertices_;
    if (n == 0)
        return {};

    if (!h_parent.mappedPtr || h_parent.size < n * sizeof(uint32_t)) {
        ctx.deallocate(h_parent);
        h_parent = ctx.pinned_allocator->allocate(n * sizeof(uint32_t), 0);
        CheckAllocation(h_parent, "cc: host parent");
    }

    ctx.barrier(d_parent, Access::Write, Access::TransferRead);
    ctx.copy(d_parent, h_parent);
    ctx.fsync();
    ctx.invalidate(h_parent);

    const uint32_t*        parent = reinterpret_cast<const uint32_t*>(h_parent.mappedPtr);
    std::vector<uint32_t>  roots(n);
    for (uint64_t i = 0; i < n; ++i) {
        uint32_t r = static_cast<uint32_t>(i);
        while (parent[r] != r)
            r = parent[r];
        roots[i] = r;
    }
    return roots;
}

uint64_t CC::component_count()
{
    const std::vector<uint32_t> roots = resolve_roots();
    uint64_t                    count = 0;
    for (uint64_t i = 0; i < roots.size(); ++i)
        count += (roots[i] == i) ? 1u : 0u;
    return count;
}

} // namespace graphs
} // namespace pictor
