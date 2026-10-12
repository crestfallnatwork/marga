#include "tc/tc_runner.h"

#include <algorithm>
#include <stdexcept>

#include <cub/device/device_scan.cuh>

#include "common/cuda_check.h"
#include "tc/tc_kernels.cuh"
#include "tc_validation.hpp"

namespace marga_abl {
namespace {

using u32 = uint32_t;
using u64 = uint64_t;

struct DeviceGraph {
    u64* row_offsets    = nullptr;
    u32* column_indices = nullptr;
    u64  n = 0, m = 0;

    void upload(const HostGraph& g)
    {
        n = g.num_vertices;
        m = g.num_edges;
        CUDA_CHECK(cudaMalloc(&row_offsets, (size_t)(n + 1) * sizeof(u64)));
        CUDA_CHECK(cudaMalloc(&column_indices, (size_t)m * sizeof(u32)));
        CUDA_CHECK(cudaMemcpy(row_offsets, g.row_offsets.data(), (size_t)(n + 1) * sizeof(u64),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(column_indices, g.column_indices.data(), (size_t)m * sizeof(u32),
                              cudaMemcpyHostToDevice));
    }

    void release()
    {
        if (row_offsets)
            cudaFree(row_offsets);
        if (column_indices)
            cudaFree(column_indices);
        row_offsets    = nullptr;
        column_indices = nullptr;
    }
};

int parse_cull(const std::string& name)
{
    if (name == "none")
        return kTcCullNone;
    if (name == "orient")
        return kTcCullOrient;
    if (name == "bounded")
        return kTcCullBounded;
    throw std::runtime_error("unknown cull mode: " + name + " (use none|orient|bounded)");
}

constexpr int kThreads = 256;

// ---------------------------------------------------------------------------
// Static arm
// ---------------------------------------------------------------------------

void launch_static(int cull, const TcLevelArgs& a, u64 batch_edges, cudaStream_t stream)
{
    const u64 blocks = (batch_edges + kThreads - 1) / kThreads;
    if (blocks == 0)
        return;
    switch (cull) {
        case kTcCullNone:
            tc_static_kernel<kTcCullNone><<<(unsigned)blocks, kThreads, 0, stream>>>(a);
            break;
        case kTcCullOrient:
            tc_static_kernel<kTcCullOrient><<<(unsigned)blocks, kThreads, 0, stream>>>(a);
            break;
        default:
            tc_static_kernel<kTcCullBounded><<<(unsigned)blocks, kThreads, 0, stream>>>(a);
            break;
    }
    CUDA_CHECK_LAUNCH();
}

// One full triangle count over the graph, batched like MARGA: per batch,
// launch the kernel, copy the accumulator back, reset it. Timed end-to-end,
// host round-trips included.
u64 run_static_once(const DeviceGraph& dg, int cull, u64 batch, u64* d_count, u64* h_count,
                    cudaStream_t stream, float* elapsed_ms)
{
    const u64 b = (batch == 0 || batch > dg.m) ? dg.m : batch;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaMemsetAsync(d_count, 0, sizeof(u64), stream));
    CUDA_CHECK(cudaEventRecord(start, stream));

    u64 total = 0;
    for (u64 edge_start = 0; edge_start < dg.m; edge_start += b) {
        const u64 edge_num = (edge_start + b <= dg.m) ? b : (dg.m - edge_start);
        TcLevelArgs a {};
        a.row_offsets    = dg.row_offsets;
        a.column_indices = dg.column_indices;
        a.num_vertices   = dg.n;
        a.num_edges      = dg.m;
        a.count          = d_count;
        a.edge_start     = edge_start;
        a.edge_num       = edge_num;
        launch_static(cull, a, edge_num, stream);

        // Per-batch readback + reset.
        CUDA_CHECK(cudaMemcpyAsync(h_count, d_count, sizeof(u64), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        total += *h_count;
        CUDA_CHECK(cudaMemsetAsync(d_count, 0, sizeof(u64), stream));
    }

    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(elapsed_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return total;
}

// ---------------------------------------------------------------------------
// Orientation pre-pass
// ---------------------------------------------------------------------------

// Two device passes (count + scan, then write) build the oriented (u < v) edge
// list from the symmetrized CSR. Edge order is preserved (no load balancing).
// Timed end-to-end, including the count readback.
DeviceGraph build_oriented(const DeviceGraph& dg, cudaStream_t stream, double* ms_out,
                           uint64_t* edges_out)
{
    DeviceGraph o;
    o.n = dg.n;

    CUDA_CHECK(cudaMalloc(&o.row_offsets, (size_t)(dg.n + 1) * sizeof(u64)));
    CUDA_CHECK(cudaMemsetAsync(o.row_offsets, 0, sizeof(u64), stream));

    u64* counts = nullptr;
    CUDA_CHECK(cudaMalloc(&counts, (size_t)dg.n * sizeof(u64)));

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0, stream));

    const int  threads = 256;
    const u64  blocks  = (dg.n + threads - 1) / threads;
    tc_orient_count_kernel<<<(unsigned)blocks, threads, 0, stream>>>(
        dg.row_offsets, dg.column_indices, counts, dg.n);
    CUDA_CHECK_LAUNCH();

    // Exclusive scan of the per-vertex counts -> oriented row offsets. The
    // scan lands in row_offsets[1..n]; row_offsets[0] stays zero.
    size_t scan_bytes = 0;
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(nullptr, scan_bytes, counts, o.row_offsets + 1,
                                             (int)dg.n, stream));
    void* scan_tmp = nullptr;
    CUDA_CHECK(cudaMalloc(&scan_tmp, scan_bytes));
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(scan_tmp, scan_bytes, counts, o.row_offsets + 1,
                                             (int)dg.n, stream));
    CUDA_CHECK(cudaFree(scan_tmp));

    CUDA_CHECK(cudaMemcpyAsync(&o.m, o.row_offsets + dg.n, sizeof(u64), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (o.m == 0 || o.m > dg.m)
        throw std::runtime_error("orientation pre-pass produced an invalid edge count");

    CUDA_CHECK(cudaMalloc(&o.column_indices, (size_t)o.m * sizeof(u32)));
    tc_orient_write_kernel<<<(unsigned)blocks, threads, 0, stream>>>(
        dg.row_offsets, dg.column_indices, o.row_offsets, o.column_indices, dg.n);
    CUDA_CHECK_LAUNCH();

    CUDA_CHECK(cudaEventRecord(t1, stream));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float prepass_ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&prepass_ms, t0, t1));
    *ms_out = (double)prepass_ms;
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    CUDA_CHECK(cudaFree(counts));
    *edges_out = o.m;
    return o;
}

} // namespace

TcAblationResult run_tc_ablation(const HostGraph& g, const TcAblationConfig& cfg)
{
    if (!g.symmetrized)
        throw std::runtime_error("TC requires a symmetrized graph");
    const int cull = parse_cull(cfg.cull);

    DeviceGraph dg;
    dg.upload(g);

    u64* d_count = nullptr;
    CUDA_CHECK(cudaMalloc(&d_count, sizeof(u64)));
    u64* h_count = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_count, sizeof(u64)));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    TcAblationResult out;
    out.min_ms = 1e30;

    // Orientation pre-pass: run the arm on the oriented (u < v) edge list.
    // The pre-pass is timed separately and is not part of the per-run average.
    DeviceGraph odg;
    if (cfg.preorient)
        odg = build_oriented(dg, stream, &out.prepass_ms, &out.oriented_edges);
    const DeviceGraph& rg = cfg.preorient ? odg : dg;

    for (int i = 0; i < cfg.warmup; ++i) {
        float ms = 0.f;
        run_static_once(rg, cull, cfg.batch, d_count, h_count, stream, &ms);
    }
    double sum = 0.0;
    for (int i = 0; i < cfg.reps; ++i) {
        float     ms = 0.f;
        const u64 c  = run_static_once(rg, cull, cfg.batch, d_count, h_count, stream, &ms);
        out.raw_count = c;
        sum += ms;
        out.min_ms = std::min(out.min_ms, (double)ms);
        out.max_ms = std::max(out.max_ms, (double)ms);
    }
    out.avg_ms = sum / (cfg.reps > 0 ? cfg.reps : 1);

    if (cfg.preorient) {
        // The oriented lists only contain >-vertex neighbors, so full-list and
        // bounded intersections both count every triangle exactly once.
        out.triangles = out.raw_count;
    } else {
        // Counts per triangle: bounded = 1 (u < v < w), orient = 3 (each of
        // the triangle's three oriented edges), none = 6 (both directions of
        // all three edges on the symmetrized graph).
        out.triangles = (cull == kTcCullBounded)  ? out.raw_count
                        : (cull == kTcCullOrient) ? out.raw_count / 3
                                                  : out.raw_count / 6;
    }

    if (cfg.validate) {
        out.validated     = true;
        out.cpu_triangles = pictor::graphs::TCValidator::run(g.row_offsets, g.column_indices);
        out.valid         = pictor::graphs::TCValidator::verify(out.triangles, out.cpu_triangles);
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFreeHost(h_count));
    CUDA_CHECK(cudaFree(d_count));
    odg.release();
    dg.release();
    return out;
}

} // namespace marga_abl
