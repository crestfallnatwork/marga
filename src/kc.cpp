// SPDX-License-Identifier: GPL-2.0-only
// Author: Himanshu Gupta <himanshu3@iisc.ac.in>

#include "kc.hpp"
#include "csr.hpp"
#include "helper.hpp"

#include "pictor_core/context/context.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <format>

static const uint32_t cMaxMeshDim = 65535;

namespace pictor {
namespace graphs {

namespace {
// Must match kc.slang.
constexpr uint32_t cGroupItems  = 128;
constexpr uint32_t cInitThreads = 256;

constexpr size_t cStateK         = 0;
constexpr size_t cStateBase      = 1;
constexpr size_t cStateCount     = 2;
constexpr size_t cStateNext      = 3;
constexpr size_t cStateSeedRead  = 4;
constexpr size_t cStateDone      = 5;
constexpr size_t cStateMode      = 6;
constexpr size_t cStateSurvivors = 7;
constexpr size_t cStateCapacity  = 8;
constexpr size_t cStateSeedSrc   = 9;
constexpr size_t cStatePhases    = 10;
constexpr size_t cStatePeak      = 11;
constexpr size_t cStatePrevMode  = 12;
constexpr size_t cStateMeshRuns  = 13;
constexpr size_t cStateSlots     = 14;

constexpr size_t cClaims       = 0;
constexpr size_t cKept         = 1;
constexpr size_t cMinDeg       = 2;
constexpr size_t cOverflow     = 3;
constexpr size_t cSeeds        = 4;
constexpr size_t cMeshRuns     = 5;
constexpr size_t cSeedMeshRuns = 6;
constexpr size_t cCounters     = 8;

using State    = std::array<uint64_t, cStateSlots>;
using Counters = std::array<uint32_t, cCounters>;

// Context::deallocate does not clear the handle; a stale one would be reused on
// the next run of the same KC object.
void drop(Context& ctx, AllocationInfo& info)
{
    ctx.deallocate(info);
    info = {};
}

void split_grid(uint64_t groups, uint32_t& x, uint32_t& y)
{
    if (groups > cMaxMeshDim) {
        x = cMaxMeshDim;
        y = static_cast<uint32_t>((groups + cMaxMeshDim - 1) / cMaxMeshDim);
    } else {
        x = static_cast<uint32_t>(groups);
        y = 1;
    }
}
} // namespace

KC::KC(Context& ctx) : ctx(ctx)
{
    work_exec.reset(WorkExec::create(ctx).val().release());
    init_exec.reset(InitExec::create(ctx).val().release());
    step_exec.reset(StepExec::create(ctx).val().release());
}

KC::~KC()
{
    release();
    drop(ctx, h_state);
    drop(ctx, h_counters);
    drop(ctx, h_core);
}

void KC::release()
{
    drop(ctx, d_state);
    drop(ctx, d_counters);
    drop(ctx, d_grid);
    drop(ctx, d_deg);
    drop(ctx, d_core);
    drop(ctx, d_removed);
    drop(ctx, d_survivors[0]);
    drop(ctx, d_survivors[1]);
    drop(ctx, d_frontier);
}

void KC::allocate(uint64_t num_vertices)
{
    frontier_cap_ = num_vertices + 1;

    d_state        = ctx.pool_allocator->allocate(cStateSlots * sizeof(uint64_t), 0);
    d_counters     = ctx.pool_allocator->allocate(cCounters * sizeof(uint32_t), 0);
    d_grid         = ctx.pool_allocator->allocate(3 * sizeof(uint32_t), 0);
    d_deg          = ctx.pool_allocator->allocate(num_vertices * sizeof(uint32_t), 0);
    d_core         = ctx.pool_allocator->allocate(num_vertices * sizeof(uint32_t), 0);
    d_removed      = ctx.pool_allocator->allocate((num_vertices + 31) / 32 * sizeof(uint32_t), 0);
    d_survivors[0] = ctx.pool_allocator->allocate(num_vertices * sizeof(uint32_t), 0);
    d_survivors[1] = ctx.pool_allocator->allocate(num_vertices * sizeof(uint32_t), 0);
    d_frontier     = ctx.pool_allocator->allocate(frontier_cap_ * sizeof(uint32_t), 0);

    const double gib = 1073741824.0;
    CheckAllocation(d_state, "kc: state");
    CheckAllocation(d_counters, "kc: counters");
    CheckAllocation(d_grid, "kc: indirect grid");
    CheckAllocation(d_deg, std::format("kc: degrees ({:.2f} GiB)", num_vertices * 4.0 / gib));
    CheckAllocation(d_core, std::format("kc: cores ({:.2f} GiB)", num_vertices * 4.0 / gib));
    CheckAllocation(d_removed,
                    std::format("kc: removed bitmap ({:.2f} GiB)", num_vertices / 8.0 / gib));
    CheckAllocation(d_survivors[0],
                    std::format("kc: survivors ({:.2f} GiB)", num_vertices * 4.0 / gib));
    CheckAllocation(d_survivors[1],
                    std::format("kc: survivors ({:.2f} GiB)", num_vertices * 4.0 / gib));
    CheckAllocation(d_frontier,
                    std::format("kc: frontier ({:.2f} GiB)", frontier_cap_ * 4.0 / gib));

    if (!h_state.mappedPtr)
        h_state = ctx.pinned_allocator->allocate(cStateSlots * sizeof(uint64_t), 0);
    if (!h_counters.mappedPtr)
        h_counters = ctx.pinned_allocator->allocate(cCounters * sizeof(uint32_t), 0);
    CheckAllocation(h_state, "kc: host state");
    CheckAllocation(h_counters, "kc: host counters");
}

void KC::initialise(Graph const& g)
{
    State state{};
    state[cStateCapacity] = frontier_cap_;
    // The init pass measured the minimum degree, so the first threshold can
    // use it as if a seed had just scanned the graph.
    state[cStateMode] = 1; // cModeSeed

    ctx.fill(d_removed, 0);
    ctx.fill(d_counters, 0);
    ctx.barrier({ResourceBarrier{&d_removed, Access::TransferWrite, Access::Read},
                 ResourceBarrier{&d_counters, Access::TransferWrite, Access::TransferWrite}});

    const uint32_t unset = 0xFFFFFFFFu;
    ctx.update(d_counters, cMinDeg * sizeof(uint32_t), sizeof(uint32_t), &unset);
    ctx.update(d_state, 0, cStateSlots * sizeof(uint64_t), state.data());
    ctx.barrier({ResourceBarrier{&d_removed, Access::TransferWrite, Access::Read},
                 ResourceBarrier{&d_counters, Access::TransferWrite, Access::Read},
                 ResourceBarrier{&d_state, Access::TransferWrite, Access::Read}});

    const uint64_t groups = (g.num_vertices + cInitThreads - 1) / cInitThreads;
    uint32_t gx, gy;
    split_grid(groups, gx, gy);
    const uint64_t stride = static_cast<uint64_t>(gx) * cInitThreads;

    WorkIn initIn = {
        .row_offsets    = reinterpret_cast<uint64_t*>(g.d_row_offsets.address),
        .column_indices = reinterpret_cast<uint32_t*>(g.d_column_indices.address),
        .state          = reinterpret_cast<uint64_t*>(d_state.address),
        .counters       = reinterpret_cast<uint32_t*>(d_counters.address),
        .deg            = reinterpret_cast<uint32_t*>(d_deg.address),
        .core           = reinterpret_cast<uint32_t*>(d_core.address),
        .removed        = reinterpret_cast<uint32_t*>(d_removed.address),
        .survivors0     = reinterpret_cast<uint32_t*>(d_survivors[0].address),
        .survivors1     = reinterpret_cast<uint32_t*>(d_survivors[1].address),
        .frontier       = reinterpret_cast<uint32_t*>(d_frontier.address),
        .num_vertices   = g.num_vertices,
        .stride         = stride,
        .probe          = 0,
    };
    init_exec->record(static_cast<uint32_t>(stride), gy, 1, initIn);

    ctx.barrier({ResourceBarrier{&d_deg, Access::Write, Access::Read},
                 ResourceBarrier{&d_core, Access::Write, Access::Read},
                 ResourceBarrier{&d_removed, Access::Write, Access::Read},
                 ResourceBarrier{&d_survivors[0], Access::Write, Access::Read},
                 ResourceBarrier{&d_counters, Access::Write, Access::Read},
                 ResourceBarrier{&d_state, Access::TransferWrite, Access::Read}});
}

uint64_t KC::run(Graph const& g, uint32_t phases_per_submit)
{
    using namespace std::chrono;

    release();
    phases_         = 0;
    frontier_peak_  = 0;
    max_core_       = 0;
    thresholds_     = 0;
    mesh_runs_      = 0;
    seed_mesh_runs_ = 0;
    probe_          = std::getenv("MARGA_KC_MESH_PROBE") != nullptr;
    num_vertices_   = g.num_vertices;
    init_ms_        = 0.0;
    loop_ms_        = 0.0;
    if (num_vertices_ == 0)
        return 0;

    const auto init_t0 = high_resolution_clock::now();
    allocate(num_vertices_);
    initialise(g);
    ctx.fsync();
    init_ms_ = duration<double, std::milli>(high_resolution_clock::now() - init_t0).count();

    const auto loop_t0 = high_resolution_clock::now();

    WorkIn workIn = {
        .row_offsets    = reinterpret_cast<uint64_t*>(g.d_row_offsets.address),
        .column_indices = reinterpret_cast<uint32_t*>(g.d_column_indices.address),
        .state          = reinterpret_cast<uint64_t*>(d_state.address),
        .counters       = reinterpret_cast<uint32_t*>(d_counters.address),
        .deg            = reinterpret_cast<uint32_t*>(d_deg.address),
        .core           = reinterpret_cast<uint32_t*>(d_core.address),
        .removed        = reinterpret_cast<uint32_t*>(d_removed.address),
        .survivors0     = reinterpret_cast<uint32_t*>(d_survivors[0].address),
        .survivors1     = reinterpret_cast<uint32_t*>(d_survivors[1].address),
        .frontier       = reinterpret_cast<uint32_t*>(d_frontier.address),
        .num_vertices   = num_vertices_,
        .stride         = 0,
        .probe          = probe_ ? 1u : 0u,
    };
    StepIn stepIn = {
        .state    = reinterpret_cast<uint64_t*>(d_state.address),
        .counters = reinterpret_cast<uint32_t*>(d_counters.address),
        .grid     = reinterpret_cast<uint32_t*>(d_grid.address),
        .probe    = probe_ ? 1u : 0u,
    };

    const uint32_t batch = std::max<uint32_t>(1, phases_per_submit);
    bool done            = false;
    const bool trace     = std::getenv("MARGA_KC_TRACE") != nullptr;

    // The init pass and its host-side transfer left these in a read state; the
    // first step writes them.
    ctx.barrier({ResourceBarrier{&d_state, Access::Read, Access::Write},
                 ResourceBarrier{&d_counters, Access::Read, Access::Write}});

    while (!done) {
        for (uint32_t i = 0; i < batch; ++i) {
            step_exec->record(1, 1, 1, stepIn);
            ctx.barrier({ResourceBarrier{&d_state, Access::Write, Access::Read},
                         ResourceBarrier{&d_counters, Access::Write, Access::Read},
                         ResourceBarrier{&d_grid, Access::Write, Access::IndirectRead}});

            work_exec->record_indirect(d_grid, workIn);
            ctx.barrier({ResourceBarrier{&d_state, Access::Read, Access::Write},
                         ResourceBarrier{&d_counters, Access::Write, Access::Read},
                         ResourceBarrier{&d_deg, Access::Write, Access::Read},
                         ResourceBarrier{&d_core, Access::Write, Access::Read},
                         ResourceBarrier{&d_removed, Access::Write, Access::Read},
                         ResourceBarrier{&d_survivors[0], Access::Write, Access::Read},
                         ResourceBarrier{&d_survivors[1], Access::Write, Access::Read},
                         ResourceBarrier{&d_frontier, Access::Write, Access::Read},
                         ResourceBarrier{&d_grid, Access::IndirectRead, Access::Write}});
        }

        ctx.barrier({ResourceBarrier{&d_state, Access::Write, Access::TransferRead},
                     ResourceBarrier{&d_state, Access::Read, Access::TransferRead},
                     ResourceBarrier{&d_counters, Access::Write, Access::TransferRead}});
        ctx.copy(d_state, h_state);
        ctx.copy(d_counters, h_counters);
        ctx.fsync();
        ctx.invalidate(h_state);
        ctx.invalidate(h_counters);
        ctx.barrier({ResourceBarrier{&d_state, Access::TransferRead, Access::Read},
                     ResourceBarrier{&d_counters, Access::TransferRead, Access::Read}});

        const auto* st = reinterpret_cast<const uint64_t*>(h_state.mappedPtr);
        const auto* ct = reinterpret_cast<const uint32_t*>(h_counters.mappedPtr);

        done            = st[cStateDone] != 0;
        phases_         = st[cStatePhases];
        frontier_peak_  = st[cStatePeak];
        thresholds_     = ct[cSeeds];
        mesh_runs_      = ct[cMeshRuns];
        seed_mesh_runs_ = ct[cSeedMeshRuns];

        if (probe_ && seed_mesh_runs_ != 0) {
            std::fprintf(stderr, "kc: %u mesh groups ran during seed phases\n", seed_mesh_runs_);
            std::exit(1);
        }

        if (trace) {
            std::fprintf(stderr,
                         "kc batch: phases=%llu k=%llu mode=%llu base=%llu count=%llu next=%llu "
                         "kept=%u min=%u seeds=%u\n",
                         static_cast<unsigned long long>(phases_),
                         static_cast<unsigned long long>(st[cStateK]),
                         static_cast<unsigned long long>(st[cStateMode]),
                         static_cast<unsigned long long>(st[cStateBase]),
                         static_cast<unsigned long long>(st[cStateCount]),
                         static_cast<unsigned long long>(st[cStateNext]), ct[cKept], ct[cMinDeg],
                         ct[cSeeds]);
        }

        if (ct[cOverflow] != 0) {
            std::fprintf(stderr, "kc: frontier overflow after %llu phases\n",
                         static_cast<unsigned long long>(phases_));
            std::exit(1);
        }
        if (phases_ > 4ull * num_vertices_ + 4096) {
            std::fprintf(stderr, "kc: phase cap reached (%llu)\n",
                         static_cast<unsigned long long>(phases_));
            std::exit(1);
        }
    }

    ctx.fsync();
    loop_ms_ = duration<double, std::milli>(high_resolution_clock::now() - loop_t0).count();
    return phases_;
}

std::vector<uint32_t> KC::copy_core()
{
    std::vector<uint32_t> core;
    if (!d_core.buffer || num_vertices_ == 0)
        return core;

    if (!h_core.mappedPtr || h_core.size < num_vertices_ * sizeof(uint32_t)) {
        drop(ctx, h_core);
        h_core = ctx.pinned_allocator->allocate(num_vertices_ * sizeof(uint32_t), 0);
        CheckAllocation(h_core, "kc: host cores");
    }

    ctx.barrier(d_core, Access::Write, Access::TransferRead);
    ctx.copy(d_core, h_core);
    ctx.fsync();
    ctx.invalidate(h_core);
    // No trailing barrier: barriers are recorded into the current command
    // buffer, and an unsubmitted one would leave the stream's fence unsignalled
    // for the next wait. Nothing writes the cores afterwards.

    const auto* src = reinterpret_cast<const uint32_t*>(h_core.mappedPtr);
    core.assign(src, src + num_vertices_);

    max_core_ = 0;
    for (uint32_t c : core)
        max_core_ = std::max(max_core_, c);
    return core;
}

} // namespace graphs
} // namespace pictor
