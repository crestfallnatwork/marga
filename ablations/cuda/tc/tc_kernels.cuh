#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include "common/cuda_check.h"

namespace marga_abl {

using u32 = uint32_t;
using u64 = uint64_t;

// Culling levels, from none to MARGA's full task-stage pruning.
enum TcCull : int {
    // Every edge (u, v): intersect the full adjacency lists; each triangle is
    // counted 6 times (both directions of its three edges).
    kTcCullNone = 0,
    // Orientation cull only: edges with u < v, full intersection; each
    // triangle is counted 3 times.
    kTcCullOrient = 1,
    // Orientation cull + bounded lists N_{>v}(u) and N_{>v}(v); each triangle
    // (u < v < w) is counted exactly once. This is MARGA's TC algorithm.
    kTcCullBounded = 2,
};

struct TcLevelArgs {
    const u64* row_offsets;    // CSR offsets, num_vertices + 1
    const u32* column_indices; // CSR adjacency (sorted, deduplicated)
    u64        num_vertices;
    u64        num_edges;
    u64*       count;          // global triangle accumulator (per-batch reset)
    u64        edge_start;     // first edge id of this batch
    u64        edge_num;       // edges in this batch
};

// First index in [lo, hi) with arr[idx] >= key.
__device__ __forceinline__ u64 tc_lower_bound(const u32* arr, u64 lo, u64 hi, u32 key)
{
    while (lo < hi) {
        const u64 mid = lo + ((hi - lo) >> 1);
        if (arr[mid] < key)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}

// First index in [lo, hi) with arr[idx] > key.
__device__ __forceinline__ u64 tc_upper_bound(const u32* arr, u64 lo, u64 hi, u32 key)
{
    while (lo < hi) {
        const u64 mid = lo + ((hi - lo) >> 1);
        if (arr[mid] > key)
            hi = mid;
        else
            lo = mid + 1;
    }
    return lo;
}

// Intersect two sorted adjacency subranges (smallest-list iteration).
__device__ __forceinline__ u32 tc_intersect(const u32* sarr, u64 s_lo, u64 s_hi,
                                            const u32* barr, u64 b_lo, u64 b_hi)
{
    u32 count = 0;
    for (u64 i = s_lo; i < s_hi; ++i) {
        const u32 w   = sarr[i];
        const u64 pos = tc_lower_bound(barr, b_lo, b_hi, w);
        if (pos < b_hi && barr[pos] == w)
            ++count;
    }
    return count;
}

// Intersect two sorted subranges, excluding the endpoints excl_a/excl_b.
// Used by the full-list cull levels.
__device__ __forceinline__ u32 tc_intersect_excl(const u32* sarr, u64 s_lo, u64 s_hi,
                                                 const u32* barr, u64 b_lo, u64 b_hi,
                                                 u32 excl_a, u32 excl_b)
{
    u32 count = 0;
    for (u64 i = s_lo; i < s_hi; ++i) {
        const u32 w = sarr[i];
        if (w == excl_a || w == excl_b)
            continue;
        const u64 pos = tc_lower_bound(barr, b_lo, b_hi, w);
        if (pos < b_hi && barr[pos] == w)
            ++count;
    }
    return count;
}

// ---------------------------------------------------------------------------
// Static arm: one thread per edge, no task/mesh decomposition. The cull level
// selects the candidate lists; intersections use smallest-list iteration.
// ---------------------------------------------------------------------------
template <int CULL>
__global__ void tc_static_kernel(TcLevelArgs a)
{
    const u64 e_limit = a.edge_start + a.edge_num;
    const u32 lane    = threadIdx.x & 31u;
    const u64 eid     = a.edge_start + (u64)blockIdx.x * blockDim.x + threadIdx.x;
    const u64 first   = eid - lane; // first edge of this warp's chunk

    // Source vertex of the warp's first edge (lane 0), broadcast to the warp.
    u32 u = 0;
    if (lane == 0 && first < e_limit) {
        u64 lo = 0, hi = a.num_vertices;
        while (lo < hi) {
            const u64 mid = lo + ((hi - lo) >> 1);
            if (a.row_offsets[mid] <= first)
                lo = mid + 1;
            else
                hi = mid;
        }
        u = (u32)(lo - 1);
    }
    u = __shfl_sync(0xffffffffu, u, 0);

    u32 local = 0;
    if (eid < e_limit) {
        while (eid >= a.row_offsets[u + 1])
            ++u;
        const u32 v = a.column_indices[eid];

        if (CULL == kTcCullNone) {
            if (u != v) {
                const u64 us = a.row_offsets[u], ue = a.row_offsets[u + 1];
                const u64 vs = a.row_offsets[v], ve = a.row_offsets[v + 1];
                if (ue - us <= ve - vs)
                    local = tc_intersect_excl(a.column_indices, us, ue, a.column_indices, vs, ve, u, v);
                else
                    local = tc_intersect_excl(a.column_indices, vs, ve, a.column_indices, us, ue, u, v);
            }
        } else if (u < v) {
            const u64 ue = a.row_offsets[u + 1];
            const u64 vs = a.row_offsets[v], ve = a.row_offsets[v + 1];
            if (CULL == kTcCullOrient) {
                const u64 us = a.row_offsets[u];
                if (ue - us <= ve - vs)
                    local = tc_intersect_excl(a.column_indices, us, ue, a.column_indices, vs, ve, u, v);
                else
                    local = tc_intersect_excl(a.column_indices, vs, ve, a.column_indices, us, ue, u, v);
            } else {
                // Bounded lists N_{>v}(u) and N_{>v}(v).
                const u64 vb  = tc_upper_bound(a.column_indices, vs, ve, v);
                const u64 n_u = ue - (eid + 1);
                const u64 n_v = ve - vb;
                if (n_u <= n_v)
                    local = tc_intersect(a.column_indices, eid + 1, ue, a.column_indices, vb, ve);
                else
                    local = tc_intersect(a.column_indices, vb, ve, a.column_indices, eid + 1, ue);
            }
        }
    }

    // Warp reduction.
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        local += __shfl_down_sync(0xffffffffu, local, off);
    if (lane == 0 && local != 0)
        atomicAdd(reinterpret_cast<unsigned long long*>(a.count), (unsigned long long)local);
}

// ---------------------------------------------------------------------------
// Orientation pre-pass: two device passes build the oriented (u < v) edge
// list from the symmetrized CSR (count + scan, then write). Edge order is
// preserved; no load balancing.
// ---------------------------------------------------------------------------

// Pass 1: count oriented neighbors (v > u) per vertex.
__global__ void tc_orient_count_kernel(const u64* __restrict__ row_offsets,
                                       const u32* __restrict__ column_indices,
                                       u64* __restrict__ counts, u64 n)
{
    const u64 u = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    if (u >= n)
        return;
    const u64 lo = tc_upper_bound(column_indices, row_offsets[u], row_offsets[u + 1], (u32)u);
    counts[u] = row_offsets[u + 1] - lo;
}

// Pass 2: copy each row's oriented suffix into the oriented edge list.
__global__ void tc_orient_write_kernel(const u64* __restrict__ row_offsets,
                                       const u32* __restrict__ column_indices,
                                       const u64* __restrict__ o_row_offsets,
                                       u32* __restrict__ o_column_indices, u64 n)
{
    const u64 u = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    if (u >= n)
        return;
    const u64 lo = tc_upper_bound(column_indices, row_offsets[u], row_offsets[u + 1], (u32)u);
    u64       dst = o_row_offsets[u];
    for (u64 k = lo; k < row_offsets[u + 1]; ++k)
        o_column_indices[dst++] = column_indices[k];
}

} // namespace marga_abl
