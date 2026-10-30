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
    prepare_exec.reset(PrepareExec::create(ctx).val().release());
}

BC::~BC()
{
    release();
    drop(ctx, h_counters);
    drop(ctx, h_delta);
}

void BC::release()
{
    drop(ctx, d_records);
    drop(ctx, d_frontier);
    drop(ctx, d_counters);
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

    d_counters = ctx.pool_allocator->allocate(2 * sizeof(uint32_t), 0);
    CheckAllocation(d_counters, "bc: counters");

    if (!h_counters.mappedPtr)
        h_counters = ctx.pinned_allocator->allocate(2 * sizeof(uint32_t), 0);
    CheckAllocation(h_counters, "bc: host counters");
}

void BC::prepare(uint32_t sources)
{
    // One pass over every (source, vertex) record: unvisited, sigma and delta
    // zero, then the source itself at distance 0 with one path.
    const uint64_t groups = (num_vertices_ + cPrepThreads - 1) / cPrepThreads;
    uint32_t       gx, gy;
    split_grid(static_cast<uint32_t>(groups), gx, gy);
    const uint64_t stride = static_cast<uint64_t>(gx) * cPrepThreads;

    ctx.fill(d_counters, 0);
    ctx.barrier(d_counters, Access::TransferWrite, Access::Write);

    prepare_exec->record(static_cast<uint32_t>(stride),
                         static_cast<uint32_t>((groups + gx - 1) / gx),
                         sources,
                         {
                             .records_u    = reinterpret_cast<uint32_t*>(d_records.address),
                             .records_f    = reinterpret_cast<float*>(d_records.address),
                             .num_vertices = num_vertices_,
                             .stride       = stride,
                             .num_sources  = sources,
                         });
    ctx.barrier(d_records, Access::Write, Access::Read);
}

void BC::seed_frontier(uint32_t first_source, uint32_t last_source)
{
    std::vector<uint64_t> seeds;
    seeds.reserve(last_source - first_source);
    for (uint32_t s = first_source; s < last_source; ++s) {
        const uint64_t src = static_cast<uint64_t>(s) * num_vertices_ / num_sources_;
        seeds.push_back((static_cast<uint64_t>(s) << 32) | src);
    }

    ctx.barrier(d_frontier, Access::Read, Access::TransferWrite);
    ctx.update(d_frontier, 0, seeds.size() * sizeof(uint64_t), seeds.data());
    ctx.barrier(d_frontier, Access::TransferWrite, Access::Read);
}

uint32_t BC::run(Graph const& g, uint32_t sources, uint32_t group)
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
    group_       = std::max<uint32_t>(1, std::min<uint32_t>(group, num_sources_));

    const auto init_t0 = high_resolution_clock::now();
    allocate(num_vertices_, num_sources_, group_);
    prepare(num_sources_);
    ctx.fsync();
    init_ms_ = duration<double, std::milli>(high_resolution_clock::now() - init_t0).count();

    for (uint32_t s0 = 0; s0 < num_sources_; s0 += group_) {
        const uint32_t        s1 = std::min(s0 + group_, num_sources_);
        std::vector<uint64_t> level_base;
        std::vector<uint64_t> level_count;

        levels_ = std::max(levels_, forward(g, s0, s1, level_base, level_count));
        backward(g, level_base, level_count);
    }

    ctx.fsync();
    return levels_;
}

uint32_t BC::forward(Graph const& g, uint32_t first_source, uint32_t last_source,
                     std::vector<uint64_t>& level_base, std::vector<uint64_t>& level_count)
{
    using namespace std::chrono;

    seed_frontier(first_source, last_source);
    // The previous group's dispatches read the record rows this group writes.
    ctx.barrier(d_records, Access::Read, Access::Write);

    const auto fwd_t0 = high_resolution_clock::now();

    uint64_t base  = 0;
    uint64_t count = last_source - first_source;
    uint32_t level = 0;

    level_base.clear();
    level_count.clear();
    level_base.push_back(base);
    level_count.push_back(count);

    while (count > 0) {
        const uint64_t next_base = base + count;
        if (next_base > frontier_cap_) {
            std::fprintf(stderr, "bc: frontier capacity %llu exceeded at level %u\n",
                         static_cast<unsigned long long>(frontier_cap_), level);
            std::exit(1);
        }

        ctx.fill(d_counters, 0);
        ctx.barrier(d_counters, Access::TransferWrite, Access::Write);

        const uint32_t groups = static_cast<uint32_t>(
            std::min<uint64_t>((count + cTaskItems - 1) / cTaskItems, 1ull << 31));
        uint32_t gx, gy;
        split_grid(groups, gx, gy);

        fwd_exec->record_raw(gx, gy, 1,
                             {
                                 .row_offsets    = reinterpret_cast<uint64_t*>(g.d_row_offsets.address),
                                 .column_indices = reinterpret_cast<uint32_t*>(g.d_column_indices.address),
                                 .records_u      = reinterpret_cast<uint32_t*>(d_records.address),
                                 .records_f      = reinterpret_cast<float*>(d_records.address),
                                 .frontier       = reinterpret_cast<uint64_t*>(d_frontier.address),
                                 .next_frontier  = reinterpret_cast<uint64_t*>(d_frontier.address),
                                 .counters       = reinterpret_cast<uint32_t*>(d_counters.address),
                                 .num_vertices   = num_vertices_,
                                 .frontier_base  = base,
                                 .frontier_count = count,
                                 .next_base      = next_base,
                                 .level          = level,
                             });

        ctx.barrier({ResourceBarrier{&d_records, Access::Write, Access::Read},
                     ResourceBarrier{&d_frontier, Access::Write, Access::Read},
                     ResourceBarrier{&d_counters, Access::Write, Access::TransferRead}});

        ctx.copy(d_counters, h_counters);
        ctx.fsync();
        ctx.invalidate(h_counters);
        ctx.barrier(d_counters, Access::TransferRead, Access::TransferWrite);

        const uint64_t produced = *reinterpret_cast<uint32_t*>(h_counters.mappedPtr);
        const uint64_t peak     = next_base + produced;
        frontier_peak_          = std::max(frontier_peak_, peak);

        if (trace_enabled())
            std::fprintf(stderr,
                         "bc fwd level %u: items=%llu produced=%llu wall=%.3f ms\n", level,
                         static_cast<unsigned long long>(count),
                         static_cast<unsigned long long>(produced),
                         duration<double, std::milli>(high_resolution_clock::now() - fwd_t0)
                             .count());

        if (peak > frontier_cap_) {
            std::fprintf(stderr, "bc: next frontier overflow (%llu > %llu)\n",
                         static_cast<unsigned long long>(peak),
                         static_cast<unsigned long long>(frontier_cap_));
            std::exit(1);
        }

        base = next_base;
        count = produced;
        ++level;
        if (count > 0) {
            level_base.push_back(base);
            level_count.push_back(count);
        }
    }

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

    for (size_t l = level_base.size(); l-- > 1;) {
        const uint64_t base  = level_base[l];
        const uint64_t count = level_count[l];
        if (count == 0)
            continue;

        uint64_t groups = (count + cBwdThreads - 1) / cBwdThreads;
        uint64_t gx     = std::min<uint64_t>(groups, cMaxMeshDim);
        uint64_t gy     = (groups + gx - 1) / gx;

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
                         });
        // This level's delta is read by the level below.
        ctx.barrier(d_records, Access::Write, Access::Read);
    }

    ctx.fsync();
    backward_ms_ += duration<double, std::milli>(high_resolution_clock::now() - bwd_t0).count();
}

std::vector<uint32_t> BC::source_list() const
{
    std::vector<uint32_t> out(num_sources_);
    for (uint32_t s = 0; s < num_sources_; ++s)
        out[s] = static_cast<uint32_t>(static_cast<uint64_t>(s) * num_vertices_ / num_sources_);
    return out;
}

std::vector<float> BC::copy_delta(uint64_t vertex_offset, uint64_t count)
{
    std::vector<float> out;
    if (!d_records.buffer || num_sources_ == 0)
        return out;
    if (vertex_offset >= num_vertices_)
        return out;
    count = std::min<uint64_t>(count, num_vertices_ - vertex_offset);
    if (count == 0)
        return out;

    const size_t bytes = static_cast<size_t>(count) * cRecordBytes;
    const size_t need  = static_cast<size_t>(num_sources_) * bytes;
    if (!h_delta.mappedPtr || h_delta.size < need) {
        drop(ctx, h_delta);
        h_delta = ctx.pinned_allocator->allocate(need, 0);
        CheckAllocation(h_delta, std::format("bc: host delta ({:.2f} MiB)", need / 1048576.0));
    }

    if (trace_enabled())
        std::fprintf(stderr, "bc copy_delta: %u sources x %llu vertices\n", num_sources_,
                     static_cast<unsigned long long>(count));

    ctx.barrier(d_records, Access::Write, Access::TransferRead);
    for (uint32_t s = 0; s < num_sources_; ++s) {
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
    out.resize(static_cast<size_t>(num_sources_) * count);
    for (uint32_t s = 0; s < num_sources_; ++s) {
        for (uint64_t v = 0; v < count; ++v) {
            const size_t slot = (static_cast<size_t>(s) * count + v) * cRecordWords;
            out[static_cast<size_t>(s) * count + v] = words[slot + 2];
        }
    }
    return out;
}

} // namespace graphs
} // namespace pictor
