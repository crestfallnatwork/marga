/* SPDX-License-Identifier: MPL-2.0 */

#include "pictor_core/algorithms/scan.hpp"
#include "pictor_core/context/context.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace pictor;

namespace {

template <typename T> std::vector<T> reference(const std::vector<T>& in, ScanKind kind)
{
    std::vector<T> out(in.size());
    T acc = 0;
    for (size_t i = 0; i < in.size(); i++) {
        if (kind == ScanKind::Exclusive) {
            out[i] = acc;
        }
        acc = T(acc + in[i]);
        if (kind == ScanKind::Inclusive) {
            out[i] = acc;
        }
    }
    return out;
}

template <typename T>
bool run_case(Context& ctx, Scan& scan, std::vector<T>& in, ScanKind kind, const std::string& label)
{
    const auto n        = static_cast<uint32_t>(in.size());
    const size_t nbytes = static_cast<size_t>(n) * sizeof(T);

    if (n == 0) {
        std::printf("  %-28s n=0       (skipped)\n", label.c_str());
        return true;
    }

    auto d_in  = ctx.pool_allocator->allocate(nbytes, alignof(T));
    auto d_out = ctx.pool_allocator->allocate(nbytes, alignof(T));
    auto d_scr = ctx.pool_allocator->allocate(scan.scratch_bytes(n, sizeof(T) * 8), 8);
    auto h_in  = ctx.pinned_allocator->allocate(nbytes, alignof(T));
    auto h_out = ctx.pinned_allocator->allocate(nbytes, alignof(T));

    std::memcpy(h_in.mappedPtr, in.data(), nbytes);
    ctx.flush(h_in);

    ctx.copy(h_in, d_in);
    ctx.barrier(d_in, Access::TransferWrite, Access::Read);

    auto res = scan.record(d_in, d_out, d_scr, n, sizeof(T) * 8, kind);
    if (!res.ok()) {
        std::printf("  %-28s FAILED to record (vulkan %d)\n", label.c_str(),
                    static_cast<int>(res.err()));
        return false;
    }

    ctx.barrier(d_out, Access::Write, Access::TransferRead);
    ctx.copy(d_out, h_out);
    ctx.fsync();
    ctx.invalidate(h_out);

    std::vector<T> got(n);
    std::memcpy(got.data(), h_out.mappedPtr, nbytes);
    std::vector<T> want = reference(in, kind);

    bool ok = true;
    for (uint32_t i = 0; i < n; i++) {
        if (got[i] != want[i]) {
            std::printf("  %-28s MISMATCH at %u: got %llu want %llu\n", label.c_str(), i,
                        static_cast<unsigned long long>(got[i]),
                        static_cast<unsigned long long>(want[i]));
            ok = false;
            break;
        }
    }

    ctx.deallocate(d_in);
    ctx.deallocate(d_out);
    ctx.deallocate(d_scr);
    ctx.deallocate(h_in);
    ctx.deallocate(h_out);

    std::printf("  %-28s n=%-8u %s\n", label.c_str(), n, ok ? "ok" : "FAILED");
    return ok;
}

template <typename T> bool run_width(Context& ctx, Scan& scan, const std::vector<uint32_t>& sizes)
{
    const std::string bits = (sizeof(T) == 8 ? "u64" : "u32");
    bool ok                = true;

    std::mt19937_64 rng(1234);
    for (ScanKind kind : {ScanKind::Exclusive, ScanKind::Inclusive}) {
        const char* kn = kind == ScanKind::Exclusive ? "exclusive" : "inclusive";
        for (uint32_t n : sizes) {
            std::vector<T> in(n);
            for (uint32_t i = 0; i < n; i++) {
                in[i] = T(rng());
            }
            ok &= run_case(ctx, scan, in, kind, bits + " " + kn);
        }
    }
    return ok;
}

} // namespace

int main()
{
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

    const std::vector<uint32_t> sizes = {0,   1,    31,   32,     33,      255,    256,
                                         257, 1000, 4096, 100000, 1000000, 8000000};

    bool ok = true;
    ok &= run_width<uint32_t>(*ctx, *scan, sizes);
    ok &= run_width<uint64_t>(*ctx, *scan, sizes);

    std::printf("\n%s\n", ok ? "all scan tests passed" : "SCAN TESTS FAILED");
    return ok ? 0 : 1;
}
