// Vulkan scan benchmark: pictor::Scan (decoupled look-back). Same synthetic sequence and sizes
// as the CUB benchmark (scan_cuda.cu) so the two can be compared directly.

#include "harness.hpp"
#include "synthetic.hpp"
#include "vulkan_harness.hpp"

#include "pictor_core/algorithms/scan.hpp"
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
    std::vector<int> bits{32, 64};
    std::vector<std::string> kinds{"exclusive", "inclusive"};
    std::vector<uint64_t> sizes{1u << 20u, 1u << 24u, 1u << 26u, 1u << 27u};
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
bool run_case(Context& ctx, Scan& scan, uint64_t n, bool exclusive, const Args& a, const char* bits)
{
    const size_t sz = sizeof(T);

    auto d_in  = ctx.pool_allocator->allocate(n * sz, alignof(T));
    auto d_out = ctx.pool_allocator->allocate(n * sz, alignof(T));
    auto d_scr = ctx.pool_allocator->allocate(
        scan.scratch_bytes(static_cast<uint32_t>(n), sizeof(T) * 8), 8);
    auto h_buf = ctx.pinned_allocator->allocate(cChunk * sz, alignof(T));
    if (!d_in.buffer || !d_out.buffer || !d_scr.buffer || !h_buf.buffer) {
        std::fprintf(stderr, "  allocation failed for n=%" PRIu64 "\n", n);
        return false;
    }

    for (uint64_t base = 0; base < n; base += cChunk) {
        uint64_t cnt = std::min<uint64_t>(cChunk, n - base);
        T* h         = static_cast<T*>(h_buf.mappedPtr);
        for (uint64_t i = 0; i < cnt; i++)
            h[i] = synthetic_value<T>(base + i);
        ctx.flush(h_buf);
        copy_range(ctx, h_buf, 0, d_in, base * sz, cnt * sz);
    }
    ctx.barrier(d_in, Access::TransferWrite, Access::Read);

    ScanKind kind = exclusive ? ScanKind::Exclusive : ScanKind::Inclusive;
    int iters =
        a.iters > 0 ? a.iters : static_cast<int>(std::clamp<uint64_t>(1000000000ull / n, 5, 200));

    double ms = measure_stream(ctx, a.warmup, iters, [&] {
        (void)scan.record(d_in, d_out, d_scr, static_cast<uint32_t>(n), sizeof(T) * 8, kind);
    });

    const char* ok = "-";
    bool good      = true;
    if (a.verify) {
        ctx.barrier(d_out, Access::Write, Access::TransferRead);
        StreamScanReference<T> ref(exclusive);
        for (uint64_t base = 0; base < n && good; base += cChunk) {
            uint64_t cnt = std::min<uint64_t>(cChunk, n - base);
            copy_range_to_host(ctx, d_out, base * sz, h_buf, cnt * sz);

            T* h = static_cast<T*>(h_buf.mappedPtr);
            for (uint64_t i = 0; i < cnt; i++) {
                T want = ref.expected(synthetic_value<T>(base + i));
                if (h[i] != want) {
                    std::fprintf(
                        stderr,
                        "  verify MISMATCH at %" PRIu64 ": got %" PRIu64 " want %" PRIu64 "\n",
                        base + i, static_cast<uint64_t>(h[i]), static_cast<uint64_t>(want));
                    good = false;
                    break;
                }
            }
        }
        ok = good ? "ok" : "FAILED";
    }

    double gib = static_cast<double>(n) * (2.0 * sz) / (1024.0 * 1024.0 * 1024.0);
    std::printf("vulkan-scan %-3s %-9s n=%-11" PRIu64 " %8.3f ms  %7.1f GiB/s  %6.2f Gelem/s  %s\n",
                bits, exclusive ? "exclusive" : "inclusive", n, ms, gib / (ms / 1000.0),
                static_cast<double>(n) / (ms * 1e6), ok);
    std::fflush(stdout);

    ctx.deallocate(d_in);
    ctx.deallocate(d_out);
    ctx.deallocate(d_scr);
    ctx.deallocate(h_buf);
    return good;
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

    auto scan_res = Scan::create(*ctx);
    if (!scan_res.ok()) {
        std::fprintf(stderr, "failed to create scan (vulkan %d)\n",
                     static_cast<int>(scan_res.err()));
        return 1;
    }
    auto scan = std::move(scan_res.val());

    auto props = ctx->physical_device.getProperties();
    std::printf("# Vulkan scan on %s\n", props.deviceName.data());

    bool ok = true;
    for (uint64_t n : a.sizes) {
        if (n == 0 || n >= (1u << 31u)) {
            std::fprintf(stderr, "skipping n=%" PRIu64 " (must be in [1, 2^31))\n", n);
            continue;
        }
        for (int bits : a.bits) {
            for (const auto& kind : a.kinds) {
                bool exclusive = (kind == "exclusive");
                if (bits == 32)
                    ok &= run_case<uint32_t>(*ctx, *scan, n, exclusive, a, "u32");
                else
                    ok &= run_case<uint64_t>(*ctx, *scan, n, exclusive, a, "u64");
            }
        }
    }
    return ok ? 0 : 1;
}
