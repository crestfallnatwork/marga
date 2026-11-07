// CUB reference benchmark for the scan primitive: cub::DeviceScan (single-pass decoupled
// look-back). Runs the same synthetic sequence and the same sizes as the Vulkan benchmark.

#include "harness.hpp"
#include "synthetic.hpp"

#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define CUDA_CHECK(expr)                                                                           \
    do {                                                                                           \
        cudaError_t _e = (expr);                                                                   \
        if (_e != cudaSuccess) {                                                                   \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__,     \
                         __LINE__);                                                                \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

using namespace pictor::bench;

namespace {

struct Args {
    std::vector<int> bits{32, 64};
    std::vector<std::string> kinds{"exclusive", "inclusive"};
    std::vector<uint64_t> sizes{1u << 20, 1u << 24, 1u << 26, 1u << 27};
    int iters   = 0;
    int warmup  = 5;
    bool verify = true;
};

Args parse(int argc, char** argv)
{
    Args a;
    ArgReader args(argc, argv);
    std::string k;
    while (args.next(k)) {
        if (k == "--bits") {
            a.bits.clear();
            for (auto& b : split(args.value("--bits")))
                a.bits.push_back(std::stoi(b));
        } else if (k == "--kinds") {
            a.kinds = split(args.value("--kinds"));
        } else if (k == "--sizes") {
            a.sizes.clear();
            for (auto& s : split(args.value("--sizes")))
                a.sizes.push_back(std::stoull(s));
        } else if (k == "--iters") {
            a.iters = std::stoi(args.value("--iters"));
        } else if (k == "--warmup") {
            a.warmup = std::stoi(args.value("--warmup"));
        } else if (k == "--no-verify") {
            a.verify = false;
        } else {
            args.unknown(k);
        }
    }
    return a;
}

template <typename T>
double time_scan(T* d_in, T* d_out, void* d_temp, size_t temp_bytes, int n, bool exclusive,
                 int iters, int warmup)
{
    auto launch = [&]() {
        if (exclusive)
            CUDA_CHECK(cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, d_in, d_out, n));
        else
            CUDA_CHECK(cub::DeviceScan::InclusiveSum(d_temp, temp_bytes, d_in, d_out, n));
    };

    for (int i = 0; i < warmup; i++)
        launch();
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<double> times;
    times.reserve(iters);
    for (int i = 0; i < iters; i++) {
        CUDA_CHECK(cudaEventRecord(start));
        launch();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times.push_back(ms);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return median(times);
}

template <typename T> bool verify_scan(T* d_out, uint64_t n, bool exclusive)
{
    const size_t chunk = 1u << 22;
    std::vector<T> h(chunk);
    StreamScanReference<T> ref(exclusive);
    for (uint64_t base = 0; base < n; base += chunk) {
        uint64_t cnt = std::min<uint64_t>(chunk, n - base);
        CUDA_CHECK(cudaMemcpy(h.data(), d_out + base, cnt * sizeof(T), cudaMemcpyDeviceToHost));
        for (uint64_t i = 0; i < cnt; i++) {
            T want = ref.expected(synthetic_value<T>(base + i));
            if (h[i] != want) {
                std::fprintf(stderr,
                             "  verify MISMATCH at %" PRIu64 ": got %" PRIu64 " want %" PRIu64 "\n",
                             base + i, static_cast<uint64_t>(h[i]), static_cast<uint64_t>(want));
                return false;
            }
        }
    }
    return true;
}

template <typename T> void run_case(const Args& a, uint64_t n, bool exclusive, const char* bits)
{
    const size_t bytes = n * sizeof(T);
    const int n_int    = static_cast<int>(n);

    T* d_in  = nullptr;
    T* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));

    const uint64_t gen_chunk = 1u << 22;
    std::vector<T> h(gen_chunk);
    for (uint64_t base = 0; base < n; base += gen_chunk) {
        uint64_t cnt = std::min<uint64_t>(gen_chunk, n - base);
        for (uint64_t i = 0; i < cnt; i++)
            h[i] = synthetic_value<T>(base + i);
        CUDA_CHECK(cudaMemcpy(d_in + base, h.data(), cnt * sizeof(T), cudaMemcpyHostToDevice));
    }

    size_t temp_bytes = 0;
    if (exclusive)
        CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, d_in, d_out, n_int));
    else
        CUDA_CHECK(cub::DeviceScan::InclusiveSum(nullptr, temp_bytes, d_in, d_out, n_int));
    void* d_temp = nullptr;
    CUDA_CHECK(cudaMalloc(&d_temp, std::max<size_t>(temp_bytes, 1)));

    int iters =
        a.iters > 0 ? a.iters : static_cast<int>(std::clamp<uint64_t>(1000000000ull / n, 5, 200));

    double ms = time_scan(d_in, d_out, d_temp, temp_bytes, n_int, exclusive, iters, a.warmup);

    const char* ok = "-";
    if (a.verify)
        ok = verify_scan(d_out, n, exclusive) ? "ok" : "FAILED";

    double gib = static_cast<double>(n) * (2.0 * sizeof(T)) / (1024.0 * 1024.0 * 1024.0);
    std::printf("cuda-cub    %-3s %-9s n=%-11" PRIu64 " %8.3f ms  %7.1f GiB/s  %6.2f Gelem/s  %s\n",
                bits, exclusive ? "exclusive" : "inclusive", n, ms, gib / (ms / 1000.0),
                static_cast<double>(n) / (ms * 1e6), ok);
    std::fflush(stdout);

    CUDA_CHECK(cudaFree(d_temp));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
}

} // namespace

int main(int argc, char** argv)
{
    Args a = parse(argc, argv);

    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    std::printf("# CUB DeviceScan on %s (%d SMs)\n", prop.name, prop.multiProcessorCount);

    for (uint64_t n : a.sizes) {
        if (n == 0 || n >= (1u << 31)) {
            std::fprintf(stderr, "skipping n=%" PRIu64 " (must be in [1, 2^31))\n", n);
            continue;
        }
        for (int bits : a.bits) {
            for (const auto& kind : a.kinds) {
                bool exclusive = (kind == "exclusive");
                if (bits == 32)
                    run_case<uint32_t>(a, n, exclusive, "u32");
                else
                    run_case<uint64_t>(a, n, exclusive, "u64");
            }
        }
    }
    return 0;
}
