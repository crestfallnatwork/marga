// Vulkan radix-sort benchmark: pictor::RadixSort (Onesweep, 32- or 64-bit keys). Same synthetic
// sequence and sizes as the CUB benchmark (sort_cuda.cu) so the two can be compared directly.

#include "harness.hpp"
#include "synthetic.hpp"
#include "vulkan_harness.hpp"

#include "pictor_core/algorithms/sort.hpp"
#include "pictor_core/context/context.hpp"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <string>
#include <vector>

using namespace pictor;
using namespace pictor::bench;

namespace {

struct Args {
    std::vector<uint64_t> sizes{1u << 20u, 1u << 22u, 1u << 24u, 1u << 26u};
    int iters   = 0;
    int warmup  = 3;
    bool verify = true;
    SortSpec spec{};         // u32 keys, ascending, whole key
    SortShape shape{};       // zero fields = pick for the device
    uint32_t value_bits = 0; // 0 = keys only
};

// 0 unsigned, 1 signed, 2 float: the shared order_key's encoding.
uint32_t order_index(SortKeyOrder order)
{
    if (order == SortKeyOrder::Signed)
        return 1;
    if (order == SortKeyOrder::Float)
        return 2;
    return 0;
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
        } else if (k == "--shape") {
            auto parts = split(args.value("--shape"));
            if (parts.size() != 2) {
                std::fprintf(stderr, "--shape wants WAVES,VPT\n");
                std::exit(1);
            }
            a.shape = SortShape{static_cast<uint32_t>(std::stoul(parts[0])),
                                static_cast<uint32_t>(std::stoul(parts[1]))};
        } else if (k == "--keys") {
            a.spec.key_bits = static_cast<uint32_t>(std::stoul(args.value("--keys")));
            if (a.spec.key_bits != 32 && a.spec.key_bits != 64) {
                std::fprintf(stderr, "--keys wants 32|64\n");
                std::exit(1);
            }
        } else if (k == "--order") {
            std::string v = args.value("--order");
            if (v == "unsigned") {
                a.spec.key_order = SortKeyOrder::Unsigned;
            } else if (v == "signed") {
                a.spec.key_order = SortKeyOrder::Signed;
            } else if (v == "float") {
                a.spec.key_order = SortKeyOrder::Float;
            } else {
                std::fprintf(stderr, "--order wants unsigned|signed|float\n");
                std::exit(1);
            }
        } else if (k == "--begin-bit") {
            a.spec.begin_bit = static_cast<uint32_t>(std::stoul(args.value("--begin-bit")));
        } else if (k == "--end-bit") {
            a.spec.end_bit = static_cast<uint32_t>(std::stoul(args.value("--end-bit")));
        } else if (k == "--values") {
            a.value_bits      = static_cast<uint32_t>(std::stoul(args.value("--values")));
            a.spec.value_bits = a.value_bits;
        } else if (k == "--descending") {
            a.spec.descending = true;
        } else if (k == "--no-verify") {
            a.verify = false;
        } else {
            args.unknown(k);
        }
    }
    return a;
}

void record(Context& ctx, RadixSort& sort, const Args& a, const AllocationInfo& d_in,
            AllocationInfo& d_out, const AllocationInfo& d_vin, AllocationInfo& d_vout,
            AllocationInfo& d_scr, uint64_t n)
{
    auto res = a.value_bits == 0 ? sort.record(d_in, d_out, d_scr, n)
                                 : sort.record(d_in, d_out, d_vin, d_vout, d_scr, n);
    if (!res.ok())
        std::fprintf(stderr, "  record failed (vulkan %d)\n", static_cast<int>(res.err()));
}

bool run_case(Context& ctx, RadixSort& sort, uint64_t n, const Args& a)
{
    const size_t sz = a.spec.key_bits / 8;

    const size_t vsz = a.value_bits / 8;

    auto d_in   = ctx.pool_allocator->allocate(n * sz, sz);
    auto d_out  = ctx.pool_allocator->allocate(n * sz, sz);
    auto d_vin  = ctx.pool_allocator->allocate(n * vsz, alignof(uint32_t));
    auto d_vout = ctx.pool_allocator->allocate(n * vsz, alignof(uint32_t));
    auto d_scr  = ctx.pool_allocator->allocate(sort.scratch_bytes(n), 8);
    auto h_buf  = ctx.pinned_allocator->allocate(cChunk * (sz + vsz), 8);
    auto h_vbuf = ctx.pinned_allocator->allocate(cChunk * vsz, alignof(uint32_t));
    if (!d_in.buffer || !d_out.buffer || !d_scr.buffer || !h_buf.buffer) {
        std::fprintf(stderr, "  allocation failed for n=%" PRIu64 "\n", n);
        return false;
    }
    if (a.value_bits != 0 && (!d_vin.buffer || !d_vout.buffer || !h_vbuf.buffer)) {
        std::fprintf(stderr, "  value allocation failed for n=%" PRIu64 "\n", n);
        return false;
    }

    uint64_t input_sum = 0;
    for (uint64_t base = 0; base < n; base += cChunk) {
        uint64_t cnt = std::min<uint64_t>(cChunk, n - base);
        if (a.spec.key_bits == 64) {
            auto* h = static_cast<uint64_t*>(h_buf.mappedPtr);
            for (uint64_t i = 0; i < cnt; i++) {
                h[i] = synthetic_value<uint64_t>(base + i);
                input_sum += h[i];
            }
        } else {
            auto* h = static_cast<uint32_t*>(h_buf.mappedPtr);
            for (uint64_t i = 0; i < cnt; i++) {
                h[i] = synthetic_value<uint32_t>(base + i);
                input_sum += h[i];
            }
        }
        ctx.flush(h_buf);
        copy_range(ctx, h_buf, 0, d_in, base * sz, cnt * sz);

        if (a.value_bits != 0) {
            // The payload is the source index, so the pairing can be checked exactly:
            // key[j] must equal synthetic_value(value[j]).
            auto* hv = a.value_bits == 32
                           ? static_cast<void*>(static_cast<uint32_t*>(h_vbuf.mappedPtr))
                           : static_cast<void*>(static_cast<uint64_t*>(h_vbuf.mappedPtr));
            for (uint64_t i = 0; i < cnt; i++) {
                if (a.value_bits == 32)
                    static_cast<uint32_t*>(hv)[i] = static_cast<uint32_t>(base + i);
                else
                    static_cast<uint64_t*>(hv)[i] = base + i;
            }
            ctx.flush(h_vbuf);
            copy_range(ctx, h_vbuf, 0, d_vin, base * vsz, cnt * vsz);
        }
    }
    ctx.barrier(d_in, Access::TransferWrite, Access::Read);
    if (a.value_bits != 0)
        ctx.barrier(d_vin, Access::TransferWrite, Access::Read);

    int iters =
        a.iters > 0 ? a.iters : static_cast<int>(std::clamp<uint64_t>(500000000ull / n, 3, 100));

    double ms = measure_stream(ctx, a.warmup, iters,
                               [&] { record(ctx, sort, a, d_in, d_out, d_vin, d_vout, d_scr, n); });

    const char* ok = "-";
    bool sorted    = true;
    bool passed    = !a.verify;
    if (a.verify) {
        // Verify the same order the kernel sorts by: the twiddled key, masked to the sorted range,
        // non-decreasing (or non-increasing when descending), plus a bit-pattern sum so a dropped
        // or duplicated key is caught.
        const uint32_t key_bits = a.spec.key_bits;
        const uint32_t end_bit  = a.spec.end_bit == 0 ? key_bits : a.spec.end_bit;
        const uint32_t span     = end_bit - a.spec.begin_bit;
        const uint64_t mask     = span == 64u ? ~0ull : ((1ull << span) - 1u);
        const uint32_t order    = order_index(a.spec.key_order);
        auto ordered            = [&](uint64_t bits) {
            return (order_key(bits, key_bits, order) >> a.spec.begin_bit) & mask;
        };
        auto key_of = [&](uint64_t i) {
            return key_bits == 64
                       ? static_cast<const uint64_t*>(h_buf.mappedPtr)[i]
                       : static_cast<uint64_t>(static_cast<const uint32_t*>(h_buf.mappedPtr)[i]);
        };

        ctx.barrier(d_out, Access::Write, Access::TransferRead);
        uint64_t prev  = 0;
        bool have_prev = false;
        uint64_t sum   = 0;
        for (uint64_t base = 0; base < n && sorted; base += cChunk) {
            uint64_t cnt = std::min<uint64_t>(cChunk, n - base);
            copy_range_to_host(ctx, d_out, base * sz, h_buf, cnt * sz);

            if (a.value_bits != 0)
                copy_range_to_host(ctx, d_vout, base * vsz, h_vbuf, cnt * vsz);

            const auto* hv = static_cast<const uint32_t*>(h_vbuf.mappedPtr);
            for (uint64_t i = 0; i < cnt; i++) {
                const uint64_t bits = key_of(i);
                uint64_t v          = ordered(bits);
                if (have_prev && (a.spec.descending ? v > prev : v < prev)) {
                    std::fprintf(stderr, "  verify NOT SORTED at %" PRIu64 "\n", base + i);
                    sorted = false;
                    break;
                }
                prev      = v;
                have_prev = true;
                sum += bits;

                if (a.value_bits == 32) {
                    uint64_t expect = key_bits == 64 ? synthetic_value<uint64_t>(hv[i])
                                                     : synthetic_value<uint32_t>(hv[i]);
                    if (bits != expect) {
                        std::fprintf(stderr, "  verify VALUE MISMATCH at %" PRIu64 "\n", base + i);
                        sorted = false;
                        break;
                    }
                } else if (a.value_bits == 64) {
                    uint64_t value  = static_cast<const uint64_t*>(h_vbuf.mappedPtr)[i];
                    uint64_t expect = key_bits == 64 ? synthetic_value<uint64_t>(value)
                                                     : synthetic_value<uint32_t>(value);
                    if (bits != expect) {
                        std::fprintf(stderr, "  verify VALUE MISMATCH at %" PRIu64 "\n", base + i);
                        sorted = false;
                        break;
                    }
                }
            }
        }
        passed = sorted && sum == input_sum;
        ok     = passed ? "ok" : "FAILED";
    }

    double mov =
        static_cast<double>(n) * (2.0 * static_cast<double>(sz)) / (1024.0 * 1024.0 * 1024.0);
    const char* key_label = a.spec.key_bits == 64 ? "u64" : "u32";
    std::printf("vulkan-sort %s n=%-11" PRIu64 " %8.3f ms  %7.1f GiB/s  %6.2f Gelem/s  %s\n",
                key_label, n, ms, mov / (ms / 1000.0), static_cast<double>(n) / (ms * 1e6), ok);
    std::fflush(stdout);

    ctx.deallocate(d_in);
    ctx.deallocate(d_out);
    ctx.deallocate(d_scr);
    ctx.deallocate(h_buf);
    if (a.value_bits != 0) {
        ctx.deallocate(d_vin);
        ctx.deallocate(d_vout);
        ctx.deallocate(h_vbuf);
    }
    return passed;
}

} // namespace

int main(int argc, char** argv)
{
    Args a = parse(argc, argv);

    auto ctx_res = Context::create();
    if (!ctx_res.ok()) {
        std::fprintf(stderr, "failed to create pictor context (vulkan %d)\n",
                     static_cast<int>(ctx_res.err()));
        return 1;
    }
    auto ctx = std::move(ctx_res.val());

    auto sort_res = a.shape.waves == 0 ? RadixSort::create(*ctx, a.spec)
                                       : RadixSort::create(*ctx, a.spec, a.shape);
    if (!sort_res.ok()) {
        std::fprintf(stderr, "failed to create radix sort (vulkan %d)\n",
                     static_cast<int>(sort_res.err()));
        return 1;
    }
    auto sort = std::move(sort_res.val());

    auto props            = ctx->physical_device.getProperties();
    const SortShape shape = sort->shape();
    const SortSpec spec   = sort->spec();
    const char* order     = "unsigned";
    if (spec.key_order == SortKeyOrder::Signed)
        order = "signed";
    else if (spec.key_order == SortKeyOrder::Float)
        order = "float";
    std::printf("# Vulkan radix sort on %s (bin kernel: %s, shape %ux%u, tile %u, keys %u, order "
                "%s, bits [%u,%u), %s, values %u)\n",
                props.deviceName.data(), "ballot", shape.waves, shape.vpt,
                shape.waves * shape.vpt * ctx->device_info.subgroup_size, spec.key_bits, order,
                spec.begin_bit, spec.end_bit, spec.descending ? "descending" : "ascending",
                spec.value_bits);

    bool ok = true;
    for (uint64_t n : a.sizes) {
        if (n == 0 || n >= (1u << 31u)) {
            std::fprintf(stderr, "skipping n=%" PRIu64 " (must be in [1, 2^31))\n", n);
            continue;
        }
        ok &= run_case(*ctx, *sort, n, a);
    }
    return ok ? 0 : 1;
}
