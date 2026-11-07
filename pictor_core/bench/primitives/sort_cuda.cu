// CUB reference benchmark for the radix sort primitive: cub::DeviceRadixSort::SortKeys over the
// same synthetic sequence and the same sizes as the Vulkan benchmark (sort_vulkan.cpp). The
// --type/--begin-bit/--end-bit/--descending flags mirror the Vulkan harness's --keys/--order/spec
// flags so the two can be compared row by row.

#include "harness.hpp"
#include "synthetic.hpp"

#include <cub/device/device_radix_sort.cuh>
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
    std::vector<uint64_t> sizes{1u << 20, 1u << 22, 1u << 24, 1u << 26};
    int iters           = 0;
    int warmup          = 3;
    bool verify         = true;
    int begin_bit       = 0;
    int end_bit         = 0; // 0 = all key bits
    bool descending     = false;
    std::string type    = "u32"; // u32 | i32 | f32 | u64 | i64 | f64
    uint32_t value_bits = 0;     // 0 = keys only, otherwise 32 or 64
};

bool is_64_bit(const std::string& type)
{
    return type == "u64" || type == "i64" || type == "f64";
}

Args parse(int argc, char** argv)
{
    Args a;
    ArgReader args(argc, argv);
    std::string k;
    while (args.next(k)) {
        if (k == "--sizes") {
            a.sizes.clear();
            for (auto& s : split(args.value("--sizes")))
                a.sizes.push_back(std::stoull(s));
        } else if (k == "--iters") {
            a.iters = std::stoi(args.value("--iters"));
        } else if (k == "--warmup") {
            a.warmup = std::stoi(args.value("--warmup"));
        } else if (k == "--type") {
            a.type = args.value("--type");
            if (a.type != "u32" && a.type != "i32" && a.type != "f32" && a.type != "u64" &&
                a.type != "i64" && a.type != "f64") {
                std::fprintf(stderr, "--type wants u32|i32|f32|u64|i64|f64\n");
                std::exit(1);
            }
        } else if (k == "--begin-bit") {
            a.begin_bit = std::stoi(args.value("--begin-bit"));
        } else if (k == "--end-bit") {
            a.end_bit = std::stoi(args.value("--end-bit"));
        } else if (k == "--values") {
            a.value_bits = static_cast<uint32_t>(std::stoul(args.value("--values")));
        } else if (k == "--descending") {
            a.descending = true;
        } else if (k == "--no-verify") {
            a.verify = false;
        } else {
            args.unknown(k);
        }
    }
    const int width = is_64_bit(a.type) ? 64 : 32;
    if (a.end_bit == 0)
        a.end_bit = width;
    if (a.begin_bit < 0 || a.begin_bit >= a.end_bit || a.end_bit > width) {
        std::fprintf(stderr,
                     "--begin-bit/--end-bit must satisfy 0 <= begin < end <= %d for --type %s\n",
                     width, a.type.c_str());
        std::exit(1);
    }
    return a;
}

int order_of(const std::string& type)
{
    if (type == "i32" || type == "i64")
        return 1;
    if (type == "f32" || type == "f64")
        return 2;
    return 0;
}

template <typename T>
double time_sort(T* d_in, T* d_out, void* d_temp, size_t temp_bytes, int n, const Args& a,
                 int iters)
{
    auto launch = [&]() {
        if (a.descending) {
            CUDA_CHECK(cub::DeviceRadixSort::SortKeysDescending(d_temp, temp_bytes, d_in, d_out, n,
                                                                a.begin_bit, a.end_bit));
        } else {
            CUDA_CHECK(cub::DeviceRadixSort::SortKeys(d_temp, temp_bytes, d_in, d_out, n,
                                                      a.begin_bit, a.end_bit));
        }
    };

    for (int i = 0; i < a.warmup; i++)
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

template <typename T> bool verify_sorted(const Args& a, T* d_out, uint64_t n, uint64_t expect_sum)
{
    const size_t chunk      = 1u << 21;
    const uint32_t key_bits = static_cast<uint32_t>(sizeof(T) * 8);
    const uint32_t order    = static_cast<uint32_t>(order_of(a.type));
    std::vector<T> h(chunk);
    const uint32_t span = static_cast<uint32_t>(a.end_bit - a.begin_bit);
    const uint64_t mask = span == 64u ? ~0ull : ((1ull << span) - 1u);

    uint64_t prev  = 0;
    bool have_prev = false;
    uint64_t sum   = 0;
    for (uint64_t base = 0; base < n; base += chunk) {
        uint64_t cnt = std::min<uint64_t>(chunk, n - base);
        CUDA_CHECK(cudaMemcpy(h.data(), d_out + base, cnt * sizeof(T), cudaMemcpyDeviceToHost));
        for (uint64_t i = 0; i < cnt; i++) {
            uint64_t bits = bits_of(h[i]);
            uint64_t v    = (order_key(bits, key_bits, order) >> a.begin_bit) & mask;
            if (have_prev && (a.descending ? v > prev : v < prev)) {
                std::fprintf(stderr, "  verify NOT SORTED at %" PRIu64 "\n", base + i);
                return false;
            }
            prev      = v;
            have_prev = true;
            sum += bits;
        }
    }
    if (sum != expect_sum) {
        std::fprintf(stderr, "  verify SUM MISMATCH (%" PRIu64 " vs %" PRIu64 ")\n", sum,
                     expect_sum);
        return false;
    }
    return true;
}

template <typename T, typename TValue> bool run_case_pairs(const Args& a, uint64_t n)
{
    const size_t kbytes = n * sizeof(T);
    const size_t vbytes = n * sizeof(TValue);

    T* d_kin       = nullptr;
    T* d_kout      = nullptr;
    TValue* d_vin  = nullptr;
    TValue* d_vout = nullptr;
    CUDA_CHECK(cudaMalloc(&d_kin, kbytes));
    CUDA_CHECK(cudaMalloc(&d_kout, kbytes));
    CUDA_CHECK(cudaMalloc(&d_vin, vbytes));
    CUDA_CHECK(cudaMalloc(&d_vout, vbytes));

    const uint64_t gen_chunk = 1u << 21;
    uint64_t input_sum       = 0;
    std::vector<T> kh(gen_chunk);
    std::vector<TValue> vh(gen_chunk);
    for (uint64_t base = 0; base < n; base += gen_chunk) {
        uint64_t cnt = std::min<uint64_t>(gen_chunk, n - base);
        for (uint64_t i = 0; i < cnt; i++) {
            kh[i] = synthetic_at<T>(base + i);
            vh[i] = static_cast<TValue>(base + i);
            input_sum += bits_of(kh[i]);
        }
        CUDA_CHECK(cudaMemcpy(d_kin + base, kh.data(), cnt * sizeof(T), cudaMemcpyHostToDevice));
        CUDA_CHECK(
            cudaMemcpy(d_vin + base, vh.data(), cnt * sizeof(TValue), cudaMemcpyHostToDevice));
    }

    auto call = [&](void* temp, size_t& bytes) {
        if (a.descending) {
            return cub::DeviceRadixSort::SortPairsDescending(
                temp, bytes, d_kin, d_kout, d_vin, d_vout, (int)n, a.begin_bit, a.end_bit);
        }
        return cub::DeviceRadixSort::SortPairs(temp, bytes, d_kin, d_kout, d_vin, d_vout, (int)n,
                                               a.begin_bit, a.end_bit);
    };

    size_t temp_bytes = 0;
    CUDA_CHECK(call(nullptr, temp_bytes));
    void* d_temp = nullptr;
    CUDA_CHECK(cudaMalloc(&d_temp, std::max<size_t>(temp_bytes, 1)));

    int iters =
        a.iters > 0 ? a.iters : static_cast<int>(std::clamp<uint64_t>(500000000ull / n, 3, 100));

    for (int i = 0; i < a.warmup; i++)
        CUDA_CHECK(call(d_temp, temp_bytes));
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<double> times;
    for (int i = 0; i < iters; i++) {
        CUDA_CHECK(cudaEventRecord(start));
        CUDA_CHECK(call(d_temp, temp_bytes));
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times.push_back(ms);
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    double ms = median(times);

    // The payload is the source index, so the pairing is exactly checkable: key[j] must equal
    // synthetic_value(value[j]). The keys are also checked for order and sum: the pairing alone
    // would accept an unsorted (identity) permutation.
    const char* ok = "-";
    bool good      = true;
    if (a.verify) {
        good = verify_sorted(a, d_kout, n, input_sum);
        std::vector<T> oh(gen_chunk);
        std::vector<TValue> voh(gen_chunk);
        for (uint64_t base = 0; base < n && good; base += gen_chunk) {
            uint64_t cnt = std::min<uint64_t>(gen_chunk, n - base);
            CUDA_CHECK(
                cudaMemcpy(oh.data(), d_kout + base, cnt * sizeof(T), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(voh.data(), d_vout + base, cnt * sizeof(TValue),
                                  cudaMemcpyDeviceToHost));
            for (uint64_t i = 0; i < cnt; i++) {
                if (oh[i] != synthetic_at<T>(static_cast<uint64_t>(voh[i]))) {
                    std::fprintf(stderr, "  verify VALUE MISMATCH at %" PRIu64 "\n", base + i);
                    good = false;
                    break;
                }
            }
        }
        ok = good ? "ok" : "FAILED";
    }

    double mov =
        static_cast<double>(n) * (2.0 * (sizeof(T) + sizeof(TValue))) / (1024.0 * 1024.0 * 1024.0);
    std::printf("cuda-cub    sortp %s n=%-10" PRIu64 " %8.3f ms  %7.1f GiB/s  %6.2f Gelem/s  %s\n",
                a.type.c_str(), n, ms, mov / (ms / 1000.0), static_cast<double>(n) / (ms * 1e6),
                ok);
    std::fflush(stdout);

    CUDA_CHECK(cudaFree(d_temp));
    CUDA_CHECK(cudaFree(d_kin));
    CUDA_CHECK(cudaFree(d_kout));
    CUDA_CHECK(cudaFree(d_vin));
    CUDA_CHECK(cudaFree(d_vout));
    return good;
}

template <typename T> bool run_case(const Args& a, uint64_t n)
{
    const size_t bytes = n * sizeof(T);
    const int n_int    = static_cast<int>(n);

    T* d_in  = nullptr;
    T* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));

    const uint64_t gen_chunk = 1u << 21;
    uint64_t input_sum       = 0;
    std::vector<T> h(gen_chunk);
    for (uint64_t base = 0; base < n; base += gen_chunk) {
        uint64_t cnt = std::min<uint64_t>(gen_chunk, n - base);
        for (uint64_t i = 0; i < cnt; i++) {
            h[i] = synthetic_at<T>(base + i);
            input_sum += bits_of(h[i]);
        }
        CUDA_CHECK(cudaMemcpy(d_in + base, h.data(), cnt * sizeof(T), cudaMemcpyHostToDevice));
    }

    size_t temp_bytes = 0;
    if (a.descending) {
        CUDA_CHECK(cub::DeviceRadixSort::SortKeysDescending(nullptr, temp_bytes, d_in, d_out, n_int,
                                                            a.begin_bit, a.end_bit));
    } else {
        CUDA_CHECK(cub::DeviceRadixSort::SortKeys(nullptr, temp_bytes, d_in, d_out, n_int,
                                                  a.begin_bit, a.end_bit));
    }
    void* d_temp = nullptr;
    CUDA_CHECK(cudaMalloc(&d_temp, std::max<size_t>(temp_bytes, 1)));

    int iters =
        a.iters > 0 ? a.iters : static_cast<int>(std::clamp<uint64_t>(500000000ull / n, 3, 100));

    double ms = time_sort(d_in, d_out, d_temp, temp_bytes, n_int, a, iters);

    const char* ok = "-";
    bool good      = true;
    if (a.verify) {
        good = verify_sorted(a, d_out, n, input_sum);
        ok   = good ? "ok" : "FAILED";
    }

    double mov = static_cast<double>(n) * (2.0 * sizeof(T)) / (1024.0 * 1024.0 * 1024.0);
    std::printf("cuda-cub    sort %s n=%-11" PRIu64 " %8.3f ms  %7.1f GiB/s  %6.2f Gelem/s  %s\n",
                a.type.c_str(), n, ms, mov / (ms / 1000.0), static_cast<double>(n) / (ms * 1e6),
                ok);
    std::fflush(stdout);

    CUDA_CHECK(cudaFree(d_temp));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return good;
}

} // namespace

int main(int argc, char** argv)
{
    Args a = parse(argc, argv);

    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    std::printf("# CUB DeviceRadixSort on %s (sm_%d%d, %d SMs, type %s, bits [%d,%d), %s)\n",
                prop.name, prop.major, prop.minor, prop.multiProcessorCount, a.type.c_str(),
                a.begin_bit, a.end_bit, a.descending ? "descending" : "ascending");

    bool ok = true;
    for (uint64_t n : a.sizes) {
        if (n == 0 || n >= (1u << 31)) {
            std::fprintf(stderr, "skipping n=%" PRIu64 " (must be in [1, 2^31))\n", n);
            continue;
        }
        const auto dispatch = [&](auto key_tag, auto value_tag) -> bool {
            using TKey   = decltype(key_tag);
            using TValue = decltype(value_tag);
            if (a.value_bits == 0)
                return run_case<TKey>(a, n);
            if (a.value_bits == 32)
                return run_case_pairs<TKey, uint32_t>(a, n);
            return run_case_pairs<TKey, uint64_t>(a, n);
        };
        if (a.type == "u32") {
            ok &= dispatch(uint32_t{}, uint32_t{});
        } else if (a.type == "i32") {
            ok &= dispatch(int32_t{}, uint32_t{});
        } else if (a.type == "f32") {
            ok &= dispatch(float{}, uint32_t{});
        } else if (a.type == "u64") {
            ok &= dispatch(uint64_t{}, uint32_t{});
        } else if (a.type == "i64") {
            ok &= dispatch(int64_t{}, uint32_t{});
        } else {
            ok &= dispatch(double{}, uint32_t{});
        }
    }
    return ok ? 0 : 1;
}
