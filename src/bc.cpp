// SPDX-License-Identifier: GPL-2.0-only
// Author: Himanshu Gupta <himanshu3@iisc.ac.in>

#include "bc.hpp"
#include "csr.hpp"
#include "helper.hpp"

#include "pictor_core/context/context.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <format>

static const uint32_t cMaxMeshDim  = 65535;
static const uint64_t cRecordBytes = 16;
static const uint64_t cRecordWords = 4;

namespace pictor {
namespace graphs {

namespace {
// Must match cItemsPerLane * WAVE_SIZE in bc.slang.
constexpr uint64_t cTaskItems   = 256;
constexpr uint32_t cBwdThreads  = 128;
constexpr uint32_t cHubThreads  = 256;
constexpr uint32_t cPrepThreads = 256;

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

// Context::deallocate does not clear the handle; a stale one would be reused on
// the next run of the same BC object.
void drop(Context& ctx, AllocationInfo& info)
{
    ctx.deallocate(info);
    info = {};
}

// Same buffer, shifted view. Copies and barriers address buffer + offset, so a
// view lets one allocation be copied in pieces.
AllocationInfo view(const AllocationInfo& base, size_t byte_offset, size_t size)
{
    AllocationInfo v = base;
    v.offset += byte_offset;
    v.address += byte_offset;
    v.size = size;
    return v;
}

bool trace_enabled()
{
    static const bool enabled = std::getenv("MARGA_BC_TRACE") != nullptr;
    return enabled;
}
} // namespace

BC::BC(Context& ctx, size_t pool_bytes) : ctx(ctx), pool_bytes_(pool_bytes)
{
    fwd_exec.reset(FwdExec::create(ctx).val().release());
    bwd_exec.reset(BwdExec::create(ctx).val().release());
    bwdm_exec.reset(BwdMeshExec::create(ctx).val().release());
    hub_exec.reset(HubExec::create(ctx).val().release());
    warp_exec.reset(WarpExec::create(ctx).val().release());
    step_exec.reset(StepExec::create(ctx).val().release());
    prepare_exec.reset(PrepareExec::create(ctx).val().release());
}

BC::~BC()
{
    release();
    drop(ctx, h_counters);
    drop(ctx, h_state);
    drop(ctx, h_levels);
    drop(ctx, h_sources);
    drop(ctx, h_delta);
}

void BC::release()
{
    drop(ctx, d_records);
    drop(ctx, d_frontier);
    drop(ctx, d_counters);
    drop(ctx, d_level_state);
    drop(ctx, d_levels);
    drop(ctx, d_grid);
    drop(ctx, d_sources);
    drop(ctx, d_hub);
}

uint32_t BC::choose_sources(uint64_t num_vertices, size_t pool_bytes)
{
    // 16 B per (source, vertex) record plus an 8 B frontier entry when all
    // sources of a group share one frontier.
    const uint64_t avail = static_cast<uint64_t>(pool_bytes) * 3 / 5;
    uint32_t       best  = 0;
    for (uint32_t k : {4u, 8u, 16u, 32u, 64u}) {
        if (k > num_vertices)
            continue;
        const uint64_t need = num_vertices * k * (cRecordBytes + sizeof(uint64_t));
        if (need <= avail)
            best = k;
    }
    return best ? best : 4u;
}

void BC::allocate(uint64_t num_vertices, uint32_t sources, uint32_t group)
{
    const uint64_t records = num_vertices * sources * cRecordBytes;
    d_records = ctx.pool_allocator->allocate(records, 0);
    CheckAllocation(d_records, std::format("bc: records ({:.2f} GiB)", records / 1073741824.0));

    frontier_cap_ = num_vertices * group;
    d_frontier    = ctx.pool_allocator->allocate(frontier_cap_ * sizeof(uint64_t), 0);
    CheckAllocation(d_frontier,
                    std::format("bc: frontier ({:.2f} GiB)",
                                frontier_cap_ * sizeof(uint64_t) / 1073741824.0));

    // Row -> source vertex of the live rows, read by the prepare and the
    // frontier seed.
    d_sources = ctx.pool_allocator->allocate(sources * sizeof(uint32_t), 0);
    CheckAllocation(d_sources, "bc: source list");
    if (!h_sources.mappedPtr || h_sources.size < sources * sizeof(uint32_t)) {
        drop(ctx, h_sources);
        h_sources = ctx.pinned_allocator->allocate(sources * sizeof(uint32_t), 0);
        CheckAllocation(h_sources, "bc: host source list");
    }

    d_counters = ctx.pool_allocator->allocate(2 * sizeof(uint32_t), 0);
    CheckAllocation(d_counters, "bc: counters");
    if (!h_counters.mappedPtr)
        h_counters = ctx.pinned_allocator->allocate(2 * sizeof(uint32_t), 0);
    CheckAllocation(h_counters, "bc: host counters");

    // [0] frontier base, [1] frontier count, [2] next base, [3] levels
    // recorded, [4] frontier capacity.
    d_level_state = ctx.pool_allocator->allocate(5 * sizeof(uint64_t), 0);
    CheckAllocation(d_level_state, "bc: level state");

    // (base, count) per level, recorded by the step shader.
    levels_capacity_ = std::min<uint64_t>(num_vertices, 1ull << 22) + 8;
    d_levels = ctx.pool_allocator->allocate(levels_capacity_ * 2 * sizeof(uint64_t), 0);
    CheckAllocation(d_levels, "bc: levels");

    d_grid = ctx.pool_allocator->allocate(4 * sizeof(uint32_t), 0);
    CheckAllocation(d_grid, "bc: indirect grid");

    // Indices of the frontier items expanded one warp per item by the hub pass.
    if (hub_threshold_ > 0) {
        d_hub = ctx.pool_allocator->allocate(frontier_cap_ * sizeof(uint32_t), 0);
        CheckAllocation(d_hub,
                        std::format("bc: hub list ({:.2f} GiB)",
                                    frontier_cap_ * sizeof(uint32_t) / 1073741824.0));
    }

    if (!h_state.mappedPtr)
        h_state = ctx.pinned_allocator->allocate(5 * sizeof(uint64_t), 0);
    CheckAllocation(h_state, "bc: host level state");
    if (!h_levels.mappedPtr || h_levels.size < levels_capacity_ * 2 * sizeof(uint64_t)) {
        drop(ctx, h_levels);
        h_levels = ctx.pinned_allocator->allocate(levels_capacity_ * 2 * sizeof(uint64_t), 0);
        CheckAllocation(h_levels, "bc: host levels");
    }

    // Frontier capacity is published to the shaders through the level state.
    ctx.barrier(d_level_state, Access::Read, Access::TransferWrite);
    ctx.update(d_level_state, 4 * sizeof(uint64_t), sizeof(uint64_t), &frontier_cap_);
    ctx.barrier(d_level_state, Access::TransferWrite, Access::Read);
}

void BC::prepare(uint32_t sources)
{
    // One pass over every (source, vertex) record: unvisited, sigma and delta
    // zero, then the source itself at distance 0 with one path.
    const uint64_t groups = (num_vertices_ + cPrepThreads - 1) / cPrepThreads;
    uint32_t       gx, gy;
    split_grid(static_cast<uint32_t>(groups), gx, gy);
    const uint64_t stride = static_cast<uint64_t>(gx) * cPrepThreads;

    prepare_exec->record(static_cast<uint32_t>(stride),
                         static_cast<uint32_t>((groups + gx - 1) / gx),
                         sources,
                         {
                             .records_u    = reinterpret_cast<uint32_t*>(d_records.address),
                             .records_f    = reinterpret_cast<float*>(d_records.address),
                             .sources      = reinterpret_cast<uint32_t*>(d_sources.address),
                             .num_vertices = num_vertices_,
                             .stride       = stride,
                         });
    ctx.barrier(d_records, Access::Write, Access::Read);
}

void BC::seed_frontier(uint32_t first_source, uint32_t last_source)
{
    const uint32_t* sources = reinterpret_cast<const uint32_t*>(h_sources.mappedPtr);
    std::vector<uint64_t> seeds;
    seeds.reserve(last_source - first_source);
    for (uint32_t s = first_source; s < last_source; ++s)
        seeds.push_back((static_cast<uint64_t>(s) << 32) | sources[s]);

    ctx.barrier(d_frontier, Access::Read, Access::TransferWrite);
    ctx.update(d_frontier, 0, seeds.size() * sizeof(uint64_t), seeds.data());
    ctx.barrier(d_frontier, Access::TransferWrite, Access::Read);
}

uint32_t BC::run(Graph const& g, uint32_t sources, uint32_t group, uint32_t levels_per_submit,
                 bool scatter_backward, uint32_t hub_degree)
{
    using namespace std::chrono;

    release();
    levels_        = 0;
    forward_ms_    = 0.0;
    backward_ms_   = 0.0;
    frontier_peak_ = 0;
    num_vertices_  = g.num_vertices;
    if (num_vertices_ == 0)
        return 0;

    num_sources_ = std::max<uint32_t>(1, std::min<uint32_t>(sources, num_vertices_));

    // Sampled sources with no out-edge contribute delta = 0 to every vertex, so
    // they are dropped before any state is allocated.
    live_sources_.clear();
    for (uint32_t s = 0; s < num_sources_; ++s) {
        const uint32_t v = static_cast<uint32_t>(static_cast<uint64_t>(s) * num_vertices_ /
                                                 num_sources_);
        if (g.row_offsets[v + 1] > g.row_offsets[v])
            live_sources_.push_back(v);
    }
    rows_ = static_cast<uint32_t>(live_sources_.size());
    group_ = std::max<uint32_t>(1, std::min<uint32_t>(group, std::max<uint32_t>(rows_, 1u)));
    levels_per_submit_ = std::max<uint32_t>(1, levels_per_submit);
    scatter_backward_  = scatter_backward;
    // The hub path pays a classify and a warp dispatch per level, so it is only
    // worth enabling on graphs with enough arcs per vertex to hold hubs.
    hub_threshold_ =
        (hub_degree > 0 && g.num_edges >= 8 * num_vertices_) ? hub_degree : 0u;

    const auto init_t0 = high_resolution_clock::now();
    if (rows_ == 0) {
        init_ms_ = duration<double, std::milli>(high_resolution_clock::now() - init_t0).count();
        return 0;
    }

    allocate(num_vertices_, rows_, group_);

    // Row -> source vertex, used by the prepare pass and the frontier seed.
    std::memcpy(h_sources.mappedPtr, live_sources_.data(), rows_ * sizeof(uint32_t));
    ctx.barrier(d_sources, Access::Read, Access::TransferWrite);
    ctx.update(d_sources, 0, rows_ * sizeof(uint32_t), live_sources_.data());
    ctx.barrier(d_sources, Access::TransferWrite, Access::Read);

    prepare(rows_);
    ctx.fsync();
    init_ms_ = duration<double, std::milli>(high_resolution_clock::now() - init_t0).count();

    for (uint32_t s0 = 0; s0 < rows_; s0 += group_) {
        const uint32_t        s1 = std::min(s0 + group_, rows_);
        std::vector<uint64_t> level_base;
        std::vector<uint64_t> level_count;

        levels_ = std::max(levels_, forward(g, s0, s1, level_base, level_count));
        if (scatter_backward_)
            backward_scatter(g, level_base, level_count);
        else
            backward(g, level_base, level_count);
    }

    ctx.fsync();
    return levels_;
}

void BC::read_levels(uint64_t recorded, std::vector<uint64_t>& level_base,
                     std::vector<uint64_t>& level_count)
{
    const uint64_t n = std::min<uint64_t>(recorded, levels_capacity_);
    const size_t   bytes = static_cast<size_t>(n) * 2 * sizeof(uint64_t);

    ctx.barrier(d_levels, Access::Write, Access::TransferRead);
    ctx.copy(view(d_levels, 0, bytes), view(h_levels, 0, bytes));
    ctx.fsync();
    ctx.invalidate(h_levels);
    ctx.barrier(d_levels, Access::TransferRead, Access::Write);

    const uint64_t* levels = reinterpret_cast<const uint64_t*>(h_levels.mappedPtr);
    level_base.resize(n);
    level_count.resize(n);
    for (uint64_t i = 0; i < n; ++i) {
        level_base[i]  = levels[2 * i];
        level_count[i] = levels[2 * i + 1];
    }

    // The last submissions may have run levels with an empty frontier; they
    // have no items, so the backward pass never visits them.
    while (level_count.size() > 1 && level_count.back() == 0) {
        level_base.pop_back();
        level_count.pop_back();
    }
}

uint32_t BC::forward(Graph const& g, uint32_t first_source, uint32_t last_source,
                     std::vector<uint64_t>& level_base, std::vector<uint64_t>& level_count)
{
    using namespace std::chrono;

    seed_frontier(first_source, last_source);
    // The previous group's dispatches read the record rows this group writes.
    ctx.barrier(d_records, Access::Read, Access::Write);

    level_base.clear();
    level_count.clear();

    uint64_t base  = 0;
    uint64_t count = last_source - first_source;
    uint32_t level = 0;

    // Levels recorded so far (device side) start at zero for every group.
    const uint64_t zero = 0;
    ctx.barrier(d_level_state, Access::Read, Access::TransferWrite);
    ctx.update(d_level_state, 3 * sizeof(uint64_t), sizeof(uint64_t), &zero);
    ctx.barrier(d_level_state, Access::TransferWrite, Access::Read);

    const auto fwd_t0 = high_resolution_clock::now();

    uint64_t recorded = 0;
    while (count > 0) {
        // The step shader consumes the pending count from counters[0] and the
        // next frontier base from state[2], so seed both for this submission.
        const uint32_t pending = static_cast<uint32_t>(count);
        const uint64_t seeded  = base;  // state[2] = this submission's first base

        ctx.barrier({ResourceBarrier{&d_level_state, Access::TransferRead, Access::TransferWrite},
                     ResourceBarrier{&d_level_state, Access::Read, Access::TransferWrite},
                     ResourceBarrier{&d_counters, Access::Read, Access::TransferWrite}});
        ctx.update(d_level_state, 2 * sizeof(uint64_t), sizeof(uint64_t), &seeded);
        ctx.update(d_counters, 0, sizeof(uint32_t), &pending);
        ctx.barrier({ResourceBarrier{&d_level_state, Access::TransferWrite, Access::Read},
                     ResourceBarrier{&d_counters, Access::TransferWrite, Access::Write}});

        for (uint32_t i = 0; i < levels_per_submit_; ++i) {
            step_exec->record(1, 1, 1,
                              {
                                  .level_state     = reinterpret_cast<uint64_t*>(d_level_state.address),
                                  .levels          = reinterpret_cast<uint64_t*>(d_levels.address),
                                  .counters        = reinterpret_cast<uint32_t*>(d_counters.address),
                                  .grid            = reinterpret_cast<uint32_t*>(d_grid.address),
                                  .levels_capacity = levels_capacity_,
                              });
            ctx.barrier({ResourceBarrier{&d_level_state, Access::Write, Access::Read},
                         ResourceBarrier{&d_grid, Access::Write, Access::IndirectRead},
                         ResourceBarrier{&d_counters, Access::Write, Access::Write}});

            fwd_exec->record_indirect(
                d_grid, {
                            .row_offsets    = reinterpret_cast<uint64_t*>(g.d_row_offsets.address),
                            .column_indices = reinterpret_cast<uint32_t*>(g.d_column_indices.address),
                            .records_u      = reinterpret_cast<uint32_t*>(d_records.address),
                            .records_f      = reinterpret_cast<float*>(d_records.address),
                            .frontier       = reinterpret_cast<uint64_t*>(d_frontier.address),
                            .level_state    = reinterpret_cast<uint64_t*>(d_level_state.address),
                            .counters       = reinterpret_cast<uint32_t*>(d_counters.address),
                            .num_vertices   = num_vertices_,
                            .level          = level + i,
                        });

            // This level's distance/sigma/frontier writes are read by the next
            // step and by the next level; the level state is about to be
            // rewritten by the next step.
            ctx.barrier({ResourceBarrier{&d_records, Access::Write, Access::Read},
                         ResourceBarrier{&d_frontier, Access::Write, Access::Read},
                         ResourceBarrier{&d_counters, Access::Write, Access::Read},
                         ResourceBarrier{&d_level_state, Access::Read, Access::Write},
                         ResourceBarrier{&d_grid, Access::IndirectRead, Access::Write}});
        }

        ctx.barrier({ResourceBarrier{&d_level_state, Access::Write, Access::TransferRead},
                     ResourceBarrier{&d_level_state, Access::Read, Access::TransferRead},
                     ResourceBarrier{&d_counters, Access::Write, Access::TransferRead}});
        ctx.copy(view(d_level_state, 0, 4 * sizeof(uint64_t)), view(h_state, 0, 4 * sizeof(uint64_t)));
        ctx.copy(view(d_counters, 0, sizeof(uint32_t)), view(h_counters, 0, sizeof(uint32_t)));
        ctx.fsync();
        ctx.invalidate(h_state);
        ctx.invalidate(h_counters);
        ctx.barrier({ResourceBarrier{&d_level_state, Access::TransferRead, Access::Read},
                     ResourceBarrier{&d_counters, Access::TransferRead, Access::Read}});

        const uint64_t* st = reinterpret_cast<const uint64_t*>(h_state.mappedPtr);
        // state[2] is the base of the level the next step will prepare, and
        // counters[0] holds that level's item count (the last level's claims).
        base     = st[2];
        count    = *reinterpret_cast<const uint32_t*>(h_counters.mappedPtr);
        recorded = st[3];
        frontier_peak_ = std::max(frontier_peak_, st[2]);

        if (trace_enabled())
            std::fprintf(stderr,
                         "bc fwd batch: levels %u..%u, next base=%llu count=%llu recorded=%llu "
                         "wall=%.3f ms\n",
                         level, level + levels_per_submit_ - 1, static_cast<unsigned long long>(base),
                         static_cast<unsigned long long>(count),
                         static_cast<unsigned long long>(recorded),
                         duration<double, std::milli>(high_resolution_clock::now() - fwd_t0)
                             .count());

        if (recorded >= levels_capacity_) {
            std::fprintf(stderr, "bc: more than %llu levels; raise the levels capacity\n",
                         static_cast<unsigned long long>(levels_capacity_));
            std::exit(1);
        }
        if (st[2] > frontier_cap_) {
            std::fprintf(stderr, "bc: frontier overflow (%llu > %llu)\n",
                         static_cast<unsigned long long>(st[2]),
                         static_cast<unsigned long long>(frontier_cap_));
            std::exit(1);
        }

        level += levels_per_submit_;
    }

    read_levels(recorded, level_base, level_count);

    ctx.fsync();
    forward_ms_ += duration<double, std::milli>(high_resolution_clock::now() - fwd_t0).count();
    return static_cast<uint32_t>(level_base.size()) - 1;
}

void BC::backward(Graph const& g, std::vector<uint64_t> const& level_base,
                  std::vector<uint64_t> const& level_count)
{
    using namespace std::chrono;

    const auto bwd_t0 = high_resolution_clock::now();
    // The forward pass read the record rows; the backward pass writes them.
    ctx.barrier(d_records, Access::Read, Access::Write);

    const bool hub = hub_threshold_ > 0 && d_hub.buffer;
    const uint32_t gather_threshold = hub ? hub_threshold_ : 0xFFFFFFFFu;

    for (size_t l = level_base.size(); l-- > 1;) {
        const uint64_t base  = level_base[l];
        const uint64_t count = level_count[l];
        if (count == 0)
            continue;

        uint64_t groups = (count + cBwdThreads - 1) / cBwdThreads;
        uint64_t gx     = std::min<uint64_t>(groups, cMaxMeshDim);
        uint64_t gy     = (groups + gx - 1) / gx;

        if (hub) {
            // Collect the level's hub items (degree >= threshold); the warp
            // pass expands each of them over one warp, the gather pass below
            // skips them and handles the rest one thread per item.
            const uint32_t zero = 0;
            ctx.barrier(d_counters, Access::Read, Access::TransferWrite);
            ctx.update(d_counters, sizeof(uint32_t), sizeof(uint32_t), &zero);
            ctx.barrier(d_counters, Access::TransferWrite, Access::Read);

            const uint64_t hgroups = (count + cHubThreads - 1) / cHubThreads;
            uint64_t       hx      = std::min<uint64_t>(hgroups, cMaxMeshDim);
            uint64_t       hy      = (hgroups + hx - 1) / hx;
            hub_exec->record(static_cast<uint32_t>(hx * cHubThreads), static_cast<uint32_t>(hy), 1,
                             {
                                 .row_offsets     = reinterpret_cast<uint64_t*>(g.d_row_offsets.address),
                                 .column_indices  = reinterpret_cast<uint32_t*>(g.d_column_indices.address),
                                 .records_u       = reinterpret_cast<uint32_t*>(d_records.address),
                                 .records_f       = reinterpret_cast<float*>(d_records.address),
                                 .frontier        = reinterpret_cast<uint64_t*>(d_frontier.address),
                                 .hub_list        = reinterpret_cast<uint32_t*>(d_hub.address),
                                 .counters        = reinterpret_cast<uint32_t*>(d_counters.address),
                                 .frontier_base   = base,
                                 .frontier_count  = count,
                                 .num_vertices    = num_vertices_,
                                 .stride          = hx * cHubThreads,
                                 .level           = static_cast<uint32_t>(l),
                                 .hub_threshold   = hub_threshold_,
                                 .wave_count      = 0,
                             });
            ctx.barrier({ResourceBarrier{&d_counters, Access::Write, Access::Read},
                         ResourceBarrier{&d_hub, Access::Write, Access::Read}});

            // A fixed warp grid walks the collected hub list on the device; the
            // list length never reaches the number of items, so the grid is
            // sized from the level rather than from the list.
            const uint32_t blocks = static_cast<uint32_t>(
                std::min<uint64_t>(std::max<uint64_t>(count / 2048, 8), 8192));
            const uint32_t warps = blocks * 4;
            warp_exec->record(blocks * cBwdThreads, 1, 1,
                              {
                                  .row_offsets     = reinterpret_cast<uint64_t*>(g.d_row_offsets.address),
                                  .column_indices  = reinterpret_cast<uint32_t*>(g.d_column_indices.address),
                                  .records_u       = reinterpret_cast<uint32_t*>(d_records.address),
                                  .records_f       = reinterpret_cast<float*>(d_records.address),
                                  .frontier        = reinterpret_cast<uint64_t*>(d_frontier.address),
                                  .hub_list        = reinterpret_cast<uint32_t*>(d_hub.address),
                                  .counters        = reinterpret_cast<uint32_t*>(d_counters.address),
                                  .frontier_base   = base,
                                  .frontier_count  = count,
                                  .num_vertices    = num_vertices_,
                                  .stride          = 0,
                                  .level           = static_cast<uint32_t>(l),
                                  .hub_threshold   = hub_threshold_,
                                  .wave_count      = warps,
                              });
        }

        if (trace_enabled())
            std::fprintf(stderr, "bc bwd level %zu: items=%llu grid=%llux%llu\n", l,
                         static_cast<unsigned long long>(count),
                         static_cast<unsigned long long>(gx), static_cast<unsigned long long>(gy));

        bwd_exec->record(static_cast<uint32_t>(gx * cBwdThreads), static_cast<uint32_t>(gy), 1,
                         {
                             .row_offsets    = reinterpret_cast<uint64_t*>(g.d_row_offsets.address),
                             .column_indices = reinterpret_cast<uint32_t*>(g.d_column_indices.address),
                             .records_u      = reinterpret_cast<uint32_t*>(d_records.address),
                             .records_f      = reinterpret_cast<float*>(d_records.address),
                             .frontier       = reinterpret_cast<uint64_t*>(d_frontier.address),
                             .frontier_base  = base,
                             .frontier_count = count,
                             .num_vertices   = num_vertices_,
                             .stride         = gx * cBwdThreads,
                             .level          = static_cast<uint32_t>(l),
                             .hub_threshold  = gather_threshold,
                         });
        // This level's delta is read by the level below.
        ctx.barrier(d_records, Access::Write, Access::Read);
    }

    ctx.fsync();
    backward_ms_ += duration<double, std::milli>(high_resolution_clock::now() - bwd_t0).count();
}

void BC::backward_scatter(Graph const& g, std::vector<uint64_t> const& level_base,
                          std::vector<uint64_t> const& level_count)
{
    using namespace std::chrono;

    const auto bwd_t0 = high_resolution_clock::now();
    ctx.barrier(d_records, Access::Read, Access::Write);

    for (size_t l = level_base.size(); l-- > 1;) {
        const uint64_t base  = level_base[l];
        const uint64_t count = level_count[l];
        if (count == 0)
            continue;

        const uint32_t groups = static_cast<uint32_t>((count + cTaskItems - 1) / cTaskItems);
        uint32_t       gx, gy;
        split_grid(groups, gx, gy);

        bwdm_exec->record_raw(gx, gy, 1,
                              {
                                  .row_offsets    = reinterpret_cast<uint64_t*>(g.d_row_offsets.address),
                                  .column_indices = reinterpret_cast<uint32_t*>(g.d_column_indices.address),
                                  .records_u      = reinterpret_cast<uint32_t*>(d_records.address),
                                  .records_f      = reinterpret_cast<float*>(d_records.address),
                                  .frontier       = reinterpret_cast<uint64_t*>(d_frontier.address),
                                  .frontier_base  = base,
                                  .frontier_count = count,
                                  .num_vertices   = num_vertices_,
                                  .stride         = 0,
                                  .level          = static_cast<uint32_t>(l),
                              });
        ctx.barrier(d_records, Access::Write, Access::Read);
    }

    ctx.fsync();
    backward_ms_ += duration<double, std::milli>(high_resolution_clock::now() - bwd_t0).count();
}

std::vector<uint32_t> BC::source_list() const
{
    return live_sources_;
}

std::vector<float> BC::copy_delta(uint64_t vertex_offset, uint64_t count)
{
    std::vector<float> out;
    if (!d_records.buffer || rows_ == 0)
        return out;
    if (vertex_offset >= num_vertices_)
        return out;
    count = std::min<uint64_t>(count, num_vertices_ - vertex_offset);
    if (count == 0)
        return out;

    const size_t bytes = static_cast<size_t>(count) * cRecordBytes;
    const size_t need  = static_cast<size_t>(rows_) * bytes;
    if (!h_delta.mappedPtr || h_delta.size < need) {
        drop(ctx, h_delta);
        h_delta = ctx.pinned_allocator->allocate(need, 0);
        CheckAllocation(h_delta, std::format("bc: host delta ({:.2f} MiB)", need / 1048576.0));
    }

    if (trace_enabled())
        std::fprintf(stderr, "bc copy_delta: %u sources x %llu vertices\n", rows_,
                     static_cast<unsigned long long>(count));

    ctx.barrier(d_records, Access::Write, Access::TransferRead);
    for (uint32_t s = 0; s < rows_; ++s) {
        const size_t src_off =
            static_cast<size_t>((static_cast<uint64_t>(s) * num_vertices_ + vertex_offset) *
                                cRecordBytes);
        ctx.copy(view(d_records, src_off, bytes), view(h_delta, static_cast<size_t>(s) * bytes, bytes));
    }
    ctx.fsync();
    ctx.invalidate(h_delta);
    // No trailing barrier: barriers are recorded into the current command
    // buffer, and leaving one un-submitted would leave the stream's fence
    // unsignalled for the next wait. Nothing writes the records afterwards.

    const float* words = reinterpret_cast<const float*>(h_delta.mappedPtr);
    out.resize(static_cast<size_t>(rows_) * count);
    for (uint32_t s = 0; s < rows_; ++s) {
        for (uint64_t v = 0; v < count; ++v) {
            const size_t slot = (static_cast<size_t>(s) * count + v) * cRecordWords;
            out[static_cast<size_t>(s) * count + v] = words[slot + 2];
        }
    }
    return out;
}

} // namespace graphs
} // namespace pictor
