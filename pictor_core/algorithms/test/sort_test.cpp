/* SPDX-License-Identifier: MPL-2.0 */

#include "pictor_core/algorithms/sort.hpp"
#include "pictor_core/context/context.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

using namespace pictor;

namespace {

bool check(bool condition, const char* what)
{
    if (!condition)
        std::fprintf(stderr, "  %-28s FAILED\n", what);
    return condition;
}

// The order-preserving key the float twiddle implements: -0 before +0, and NaNs by bit pattern,
// which is what the shader sorts by.
uint32_t float_order_key(float value)
{
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & 0x80000000u) != 0u ? ~bits : (bits | 0x80000000u);
}

uint64_t float_order_key(double value)
{
    uint64_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & 0x8000000000000000ull) != 0ull ? ~bits : (bits | 0x8000000000000000ull);
}

template <typename T, typename TLess = std::less<T>, typename TEqual = std::equal_to<T>>
bool run_case(Context& ctx, RadixSort& sort, const std::vector<T>& in, const std::string& label,
              TLess less = TLess{}, uint32_t strip_items = 0, TEqual equal = TEqual{})
{
    const auto n      = static_cast<uint32_t>(in.size());
    const auto nbytes = static_cast<size_t>(n) * sizeof(T);

    if (n == 0) {
        std::printf("  %-28s n=0       (skipped)\n", label.c_str());
        return true;
    }

    auto d_in  = ctx.pool_allocator->allocate(nbytes, alignof(T));
    auto d_out = ctx.pool_allocator->allocate(nbytes, alignof(T));
    auto d_scr = ctx.pool_allocator->allocate(sort.scratch_bytes(n, strip_items), 8);
    auto h_in  = ctx.pinned_allocator->allocate(nbytes, alignof(T));
    auto h_out = ctx.pinned_allocator->allocate(nbytes, alignof(T));

    std::memcpy(h_in.mappedPtr, in.data(), nbytes);
    ctx.flush(h_in);
    ctx.copy(h_in, d_in);
    ctx.barrier(d_in, Access::TransferWrite, Access::Read);

    auto res = sort.record(d_in, d_out, d_scr, n, strip_items);
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
    std::vector<T> want = in;
    std::stable_sort(want.begin(), want.end(), less);

    bool ok = true;
    for (uint32_t i = 0; i < n; i++) {
        if (!equal(got[i], want[i])) {
            std::printf("  %-28s MISMATCH at %u (n=%u strip=%u): got %g want %g\n", label.c_str(),
                        i, n, strip_items, static_cast<double>(got[i]),
                        static_cast<double>(want[i]));
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

template <typename TKey, typename TValue, typename TLess = std::nullptr_t>
bool run_pairs_case(Context& ctx, RadixSort& sort, const std::vector<TKey>& keys,
                    const std::vector<TValue>& values, const std::string& label,
                    uint32_t strip_items = 0, bool descending = false, TLess less = TLess{})
{
    const auto n = static_cast<uint32_t>(keys.size());
    if (n == 0) {
        std::printf("  %-28s n=0       (skipped)\n", label.c_str());
        return true;
    }

    const size_t key_bytes = static_cast<size_t>(n) * sizeof(TKey);
    const size_t val_bytes = static_cast<size_t>(n) * sizeof(TValue);

    auto d_kin  = ctx.pool_allocator->allocate(key_bytes, alignof(TKey));
    auto d_kout = ctx.pool_allocator->allocate(key_bytes, alignof(TKey));
    auto d_vin  = ctx.pool_allocator->allocate(val_bytes, alignof(TValue));
    auto d_vout = ctx.pool_allocator->allocate(val_bytes, alignof(TValue));
    auto d_scr  = ctx.pool_allocator->allocate(sort.scratch_bytes(n, strip_items), 8);
    auto h_keys = ctx.pinned_allocator->allocate(key_bytes, alignof(uint64_t));
    auto h_vals = ctx.pinned_allocator->allocate(val_bytes, alignof(uint64_t));

    // One host buffer per array: sharing one would let the second memcpy clobber the first before
    // its device copy has run.
    auto upload = [&](const void* src, size_t bytes, AllocationInfo& dst, AllocationInfo& host) {
        std::memcpy(host.mappedPtr, src, bytes);
        host.size = bytes;
        ctx.flush(host);
        AllocationInfo copy = dst;
        copy.size           = bytes;
        ctx.copy(host, copy);
        ctx.barrier(copy, Access::TransferWrite, Access::Read);
    };
    upload(keys.data(), key_bytes, d_kin, h_keys);
    upload(values.data(), val_bytes, d_vin, h_vals);

    auto res = sort.record(d_kin, d_kout, d_vin, d_vout, d_scr, n, strip_items);
    if (!res.ok()) {
        std::printf("  %-28s FAILED to record (vulkan %d)\n", label.c_str(),
                    static_cast<int>(res.err()));
        return false;
    }

    ctx.barrier(d_kout, Access::Write, Access::TransferRead);
    ctx.barrier(d_vout, Access::Write, Access::TransferRead);

    std::vector<TKey> got_keys(n);
    std::vector<TValue> got_values(n);
    auto download = [&](AllocationInfo& src, void* out, size_t bytes, AllocationInfo& host) {
        AllocationInfo copy = host;
        copy.size           = bytes;
        ctx.copy(src, copy);
        ctx.fsync();
        ctx.invalidate(copy);
        std::memcpy(out, copy.mappedPtr, bytes);
    };
    download(d_kout, got_keys.data(), key_bytes, h_keys);
    download(d_vout, got_values.data(), val_bytes, h_vals);

    // The reference is a stable sort of the positions, which pins both the order and the payload.
    std::vector<uint32_t> order(n);
    for (uint32_t i = 0; i < n; i++)
        order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        if constexpr (std::is_same_v<TLess, std::nullptr_t>)
            return descending ? keys[b] < keys[a] : keys[a] < keys[b];
        else
            return less(keys[a], keys[b]);
    });

    bool ok = true;
    for (uint32_t i = 0; i < n && ok; i++) {
        if (got_keys[i] != keys[order[i]] || got_values[i] != values[order[i]]) {
            std::printf("  %-28s MISMATCH at %u (n=%u strip=%u): key %g/%g value %g/%g\n",
                        label.c_str(), i, n, strip_items, static_cast<double>(got_keys[i]),
                        static_cast<double>(keys[order[i]]), static_cast<double>(got_values[i]),
                        static_cast<double>(values[order[i]]));
            ok = false;
        }
    }

    ctx.deallocate(d_kin);
    ctx.deallocate(d_kout);
    ctx.deallocate(d_vin);
    ctx.deallocate(d_vout);
    ctx.deallocate(d_scr);
    ctx.deallocate(h_keys);
    ctx.deallocate(h_vals);

    std::printf("  %-28s n=%-8u %s\n", label.c_str(), n, ok ? "ok" : "FAILED");
    return ok;
}

template <typename T = uint32_t> std::vector<T> random_keys(std::mt19937_64& rng, uint32_t n)
{
    std::vector<T> keys(n);
    for (T& key : keys)
        key = static_cast<T>(rng());
    return keys;
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

    auto sort_res = RadixSort::create(*ctx);
    if (!sort_res.ok()) {
        std::fprintf(stderr, "failed to create radix sort (vulkan %d)\n",
                     static_cast<int>(sort_res.err()));
        return 1;
    }
    auto sort = std::move(sort_res.val());

    bool ok = true;

    // The shape the tuner picked must be one the device can run, and the cost model the tuner
    // consults must track the bin kernel's declared groupshared arrays: the pinned values are
    // the declared-array sums for a 32-lane subgroup.
    const SortShape shape = sort->shape();
    std::printf("  %-28s %ux%u (tile %u)\n", "shape", shape.waves, shape.vpt,
                shape.waves * ctx->device_info.subgroup_size * shape.vpt);
    const SortSpec keys_only{};
    ok &= check(RadixSort::config_fits(ctx->device_info, keys_only, shape), "default shape fits");
    if (ctx->device_info.subgroup_size == 32) {
        ok &= check(RadixSort::shared_bytes(ctx->device_info, keys_only, {8, 16}) == 30788,
                    "shared(8, 16)");
        ok &= check(RadixSort::shared_bytes(ctx->device_info, keys_only, {12, 16}) == 43108,
                    "shared(12, 16)");
        ok &= check(RadixSort::shared_bytes(ctx->device_info, keys_only, {12, 18}) == 46180,
                    "shared(12, 18)");
        ok &= check(RadixSort::shared_bytes(ctx->device_info, keys_only, {12, 20}) == 49252,
                    "shared(12, 20)");
        // 12x20 exceeds the 49152 B on the dev GPUs (and returned VK_ERROR_DEVICE_LOST once);
        // unless the device is roomy it must be rejected.
        ok &= check(!RadixSort::config_fits(ctx->device_info, keys_only, {12, 20}) ||
                        ctx->device_info.maxComputeSharedMemorySize >= 49252,
                    "over-budget shape rejected");
        // A payload is staged next to its key, so it costs shared memory like another key.
        ok &= check(RadixSort::shared_bytes(ctx->device_info, {.value_bits = 32}, {8, 16}) == 47172,
                    "shared pairs(8, 16)");
        ok &= check(RadixSort::shared_bytes(ctx->device_info, {.value_bits = 64}, {8, 16}) == 63556,
                    "shared pairs64(8, 16)");
        // 64-bit keys widen the staged key array, and the tile shrinks to stay in budget.
        const SortSpec keys64{.key_bits = 64};
        ok &= check(RadixSort::shared_bytes(ctx->device_info, keys64, {8, 16}) == 47172,
                    "shared u64(8, 16)");
        ok &= check(RadixSort::shared_bytes(ctx->device_info, {.key_bits = 64, .value_bits = 64},
                                            {8, 16}) == 79940,
                    "shared u64 pairs64(8, 16)");
    }
    ok &= check(RadixSort::shared_bytes(ctx->device_info, {.value_bits = 64}, {8, 16}) >
                    RadixSort::shared_bytes(ctx->device_info, keys_only, {8, 16}),
                "a payload costs shared");
    ok &= check(RadixSort::shared_bytes(ctx->device_info, {.key_bits = 64}, {8, 16}) >
                    RadixSort::shared_bytes(ctx->device_info, keys_only, {8, 16}),
                "64-bit keys cost shared");

    // The rank loop's ballots cover lanes 0-31 only, so a wider subgroup must be rejected even
    // though the shape fits the device's limits.
    {
        DeviceInfo wide    = ctx->device_info;
        wide.subgroup_size = 64;
        ok &= check(!RadixSort::config_fits(wide, keys_only, shape), "subgroup > 32 rejected");
    }

    {
        // The clamp keeps a shape runnable on a device smaller than the one measured: shrink to
        // the shared-memory budget, never below one digit bin per thread.
        DeviceInfo small                 = ctx->device_info;
        small.maxComputeSharedMemorySize = 20480;
        const SortShape fitted           = RadixSort::fit_to_device(small, keys_only, {12, 18});
        ok &= check(RadixSort::config_fits(small, keys_only, fitted), "clamp fits a 20 KiB budget");

        small.maxComputeSharedMemorySize = 8192;
        const SortShape minimum          = RadixSort::fit_to_device(small, keys_only, {12, 18});
        const uint32_t min_waves = (kSortRadix + small.subgroup_size - 1) / small.subgroup_size;
        ok &= check(minimum.waves == min_waves && minimum.vpt == 1, "clamp stops at the minimum");
    }

    // An explicit shape compiles its own pipelines and must sort identically; on either dev GPU
    // this differs from the shape the default picked.
    const SortShape explicit_shape{shape.waves == 8u ? 12u : 8u, 16u};
    auto alt_res = RadixSort::create(*ctx, SortSpec{}, explicit_shape);
    if (!alt_res.ok()) {
        std::fprintf(stderr, "failed to create radix sort with shape %ux%u (vulkan %d)\n",
                     explicit_shape.waves, explicit_shape.vpt, static_cast<int>(alt_res.err()));
        return 1;
    }
    auto alt = std::move(alt_res.val());
    {
        std::mt19937_64 rng(99);
        ok &= run_case(*ctx, *alt, random_keys(rng, 4096), "u32 explicit shape");
    }

    std::mt19937_64 rng(1234);
    const std::vector<uint32_t> sizes = {0,   1,    31,   32,   33,   255,   256,
                                         257, 2047, 2048, 2049, 4096, 100000};

    for (uint32_t n : sizes)
        ok &= run_case(*ctx, *sort, random_keys(rng, n), "u32 random");

    for (uint32_t n : {1000u, 100000u}) {
        std::vector<uint32_t> few(n);
        for (uint32_t& v : few)
            v = static_cast<uint32_t>(rng() % 8u);
        ok &= run_case(*ctx, *sort, few, "u32 few-distinct");

        std::vector<uint32_t> eq(n, 7u);
        ok &= run_case(*ctx, *sort, eq, "u32 all-equal");

        std::vector<uint32_t> rev(n);
        for (uint32_t i = 0; i < n; i++)
            rev[i] = n - i;
        ok &= run_case(*ctx, *sort, rev, "u32 reverse");
    }

    // Force the strip loop: a small strip length makes the existing sizes span many strips,
    // so the per-strip carry (bins_out -> next bins_in) is exercised.
    for (uint32_t strip : {1u << 12u, 1u << 14u, 1u << 16u}) {
        for (uint32_t n : {50000u, 100000u}) {
            std::vector<uint32_t> in(n);
            for (uint32_t& v : in)
                v = static_cast<uint32_t>(rng() % 1000u);
            ok &= run_case(*ctx, *sort, in, "u32 multi-strip", std::less<uint32_t>{}, strip);
        }
    }

    // Descending sorts the same key bits the other way around, and stays stable.
    {
        auto res = RadixSort::create(*ctx, SortSpec{.descending = true});
        if (!res.ok()) {
            std::fprintf(stderr, "failed to create descending sort (vulkan %d)\n",
                         static_cast<int>(res.err()));
            return 1;
        }
        auto desc = std::move(res.val());
        for (uint32_t n : {1000u, 100000u}) {
            std::vector<uint32_t> in(n);
            for (uint32_t& v : in)
                v = static_cast<uint32_t>(rng() % 1000u);
            ok &= run_case(*ctx, *desc, in, "u32 descending", std::greater<uint32_t>{});
        }
        ok &= run_case(*ctx, *desc, random_keys(rng, 100000), "u32 descending random",
                       std::greater<uint32_t>{});
    }

    // Signed keys sort by value, via the sign-bit flip.
    {
        auto res = RadixSort::create(*ctx, SortSpec{.key_order = SortKeyOrder::Signed});
        if (!res.ok()) {
            std::fprintf(stderr, "failed to create signed sort (vulkan %d)\n",
                         static_cast<int>(res.err()));
            return 1;
        }
        auto isort = std::move(res.val());
        std::vector<int32_t> in(100000);
        for (int32_t& v : in)
            v = static_cast<int32_t>(rng());
        ok &= run_case(*ctx, *isort, in, "i32 signed");
    }

    // Floats sort by value (the shader's documented bit-pattern order: -0 < +0, NaNs by bits).
    {
        auto res = RadixSort::create(*ctx, SortSpec{.key_order = SortKeyOrder::Float});
        if (!res.ok()) {
            std::fprintf(stderr, "failed to create float sort (vulkan %d)\n",
                         static_cast<int>(res.err()));
            return 1;
        }
        auto fsort       = std::move(res.val());
        auto float_less  = [](float a, float b) { return float_order_key(a) < float_order_key(b); };
        auto float_equal = [](float a, float b) {
            return float_order_key(a) == float_order_key(b);
        };
        std::vector<float> in(100000);
        for (float& v : in)
            v = static_cast<float>(static_cast<int64_t>(rng() % 2000001) - 1000000) / 1000.0f;
        // The order key is a bijection on bit patterns, so the equal also catches -0/+0 and NaN.
        for (float special :
             {-2.5f, -0.0f, 0.0f, std::numeric_limits<float>::infinity(),
              -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN(),
              std::numeric_limits<float>::denorm_min()})
            in.push_back(special);
        ok &= run_case(*ctx, *fsort, in, "f32 float", float_less, 0, float_equal);
    }

    // A partial range sorts by those bits only: bits [8, 24) here.
    {
        auto res = RadixSort::create(*ctx, SortSpec{.begin_bit = 8, .end_bit = 24});
        if (!res.ok()) {
            std::fprintf(stderr, "failed to create range sort (vulkan %d)\n",
                         static_cast<int>(res.err()));
            return 1;
        }
        auto range_sort  = std::move(res.val());
        auto masked_less = [](uint32_t a, uint32_t b) {
            return ((a >> 8u) & 0xFFFFu) < ((b >> 8u) & 0xFFFFu);
        };
        std::vector<uint32_t> in(100000);
        for (uint32_t& v : in)
            v = static_cast<uint32_t>(rng() & 0x00FFFFFFu);
        ok &= run_case(*ctx, *range_sort, in, "u32 bits [8,24)", masked_less);
    }

    // Odd digit-place counts: the pass buffers alternate, and the last pass must land in the
    // caller's outputs (a 1-, 3-, 5- or 7-place sort used to leave the result in the scratch).
    {
        struct RangeCase {
            uint32_t key_bits;
            uint32_t begin_bit;
            uint32_t end_bit;
            bool descending;
            const char* label;
        };
        const std::array cases = {
            RangeCase{32, 8, 32, false, "u32 bits [8,32)"},
            RangeCase{32, 16, 24, false, "u32 bits [16,24)"},
            RangeCase{32, 8, 32, true, "u32 bits [8,32) desc"},
            RangeCase{64, 63, 64, false, "u64 bits [63,64)"},
            RangeCase{64, 47, 64, false, "u64 bits [47,64)"},
            RangeCase{64, 47, 64, true, "u64 bits [47,64) desc"},
        };
        for (const auto& c : cases) {
            auto res = RadixSort::create(*ctx, SortSpec{.key_bits   = c.key_bits,
                                                        .begin_bit  = c.begin_bit,
                                                        .end_bit    = c.end_bit,
                                                        .descending = c.descending});
            if (!res.ok()) {
                std::fprintf(stderr, "failed to create range sort (vulkan %d)\n",
                             static_cast<int>(res.err()));
                return 1;
            }
            auto rs = std::move(res.val());
            if (c.key_bits == 64) {
                const uint32_t begin = c.begin_bit;
                const uint64_t mask  = (1ull << (c.end_bit - c.begin_bit)) - 1ull;
                const bool desc      = c.descending;
                auto less            = [begin, mask, desc](uint64_t a, uint64_t b) {
                    const uint64_t ka = (a >> begin) & mask;
                    const uint64_t kb = (b >> begin) & mask;
                    return desc ? kb < ka : ka < kb;
                };
                for (uint32_t n : {2049u, 100000u}) {
                    std::vector<uint64_t> in(n);
                    for (uint64_t& v : in)
                        v = rng();
                    ok &= run_case(*ctx, *rs, in, c.label, less);
                }
            } else {
                const uint32_t begin = c.begin_bit;
                const uint32_t mask  = (1u << (c.end_bit - c.begin_bit)) - 1u;
                const bool desc      = c.descending;
                auto less            = [begin, mask, desc](uint32_t a, uint32_t b) {
                    const uint32_t ka = (a >> begin) & mask;
                    const uint32_t kb = (b >> begin) & mask;
                    return desc ? kb < ka : ka < kb;
                };
                std::vector<uint32_t> in(100000);
                for (uint32_t& v : in)
                    v = rng();
                ok &= run_case(*ctx, *rs, in, c.label, less);
            }
        }

        // A payload with an odd range: both the keys and the values must land in the outputs.
        auto pres = RadixSort::create(
            *ctx, SortSpec{.key_bits = 64, .begin_bit = 63, .end_bit = 64, .value_bits = 32});
        if (!pres.ok()) {
            std::fprintf(stderr, "failed to create odd-range pairs sort (vulkan %d)\n",
                         static_cast<int>(pres.err()));
            return 1;
        }
        auto rp           = std::move(pres.val());
        auto top_bit_less = [](uint64_t a, uint64_t b) { return (a >> 63u) < (b >> 63u); };
        std::vector<uint64_t> pkeys(2049);
        for (uint64_t& v : pkeys)
            v = rng();
        std::vector<uint32_t> pvalues(pkeys.size());
        for (uint32_t i = 0; i < pvalues.size(); i++)
            pvalues[i] = i;
        ok &= run_pairs_case(*ctx, *rp, pkeys, pvalues, "u64/u32 bits [63,64)", 0, false,
                             top_bit_less);
    }

    // 64-bit keys: the staged key array doubles, so the default create shrinks the shape to stay
    // within the shared budget, and every entry point ranks the wider key.
    {
        auto res = RadixSort::create(*ctx, SortSpec{.key_bits = 64});
        if (!res.ok()) {
            std::fprintf(stderr, "failed to create u64 sort (vulkan %d)\n",
                         static_cast<int>(res.err()));
            return 1;
        }
        auto sort64 = std::move(res.val());

        ok &= check(sort64->spec().key_bits == 64, "u64 spec recorded");
        ok &= check(RadixSort::config_fits(ctx->device_info, sort64->spec(), sort64->shape()),
                    "u64 shape fits");
        // A 64-bit key halves the items per thread of the same device's 32-bit shape.
        ok &= check(sort64->shape().waves == shape.waves && sort64->shape().vpt <= shape.vpt,
                    "u64 shape scales with the key");

        for (uint32_t n : {1u, 33u, 2049u, 4096u, 100000u})
            ok &= run_case(*ctx, *sort64, random_keys<uint64_t>(rng, n), "u64 random");

        std::vector<uint64_t> few(100000);
        for (uint64_t& v : few)
            v = rng() % 8u;
        ok &= run_case(*ctx, *sort64, few, "u64 few-distinct");

        std::vector<uint64_t> eq(100000, ~0ull);
        ok &= run_case(*ctx, *sort64, eq, "u64 all-equal");

        std::vector<uint64_t> rev(100000);
        for (uint64_t i = 0; i < rev.size(); i++)
            rev[i] = rev.size() - i;
        ok &= run_case(*ctx, *sort64, rev, "u64 reverse");

        std::vector<uint64_t> strip_in(100000);
        for (uint64_t& v : strip_in)
            v = rng() % 1000u;
        ok &=
            run_case(*ctx, *sort64, strip_in, "u64 multi-strip", std::less<uint64_t>{}, 1u << 12u);
    }

    // The 64-bit descending filler must land in the last bin of every place too.
    {
        auto res = RadixSort::create(*ctx, SortSpec{.key_bits = 64, .descending = true});
        if (!res.ok()) {
            std::fprintf(stderr, "failed to create u64 descending sort (vulkan %d)\n",
                         static_cast<int>(res.err()));
            return 1;
        }
        auto desc64 = std::move(res.val());
        ok &= run_case(*ctx, *desc64, random_keys<uint64_t>(rng, 100000), "u64 descending",
                       std::greater<uint64_t>{});
        std::vector<uint64_t> in(50000);
        for (uint64_t& v : in)
            v = rng() % 16u;
        ok &= run_case(*ctx, *desc64, in, "u64 descending ties", std::greater<uint64_t>{});
    }

    // Signed and float keys widen to 64 bits the same way.
    {
        auto res =
            RadixSort::create(*ctx, SortSpec{.key_bits = 64, .key_order = SortKeyOrder::Signed});
        if (!res.ok()) {
            std::fprintf(stderr, "failed to create i64 sort (vulkan %d)\n",
                         static_cast<int>(res.err()));
            return 1;
        }
        auto isort64 = std::move(res.val());
        std::vector<int64_t> in(100000);
        for (int64_t& v : in)
            v = static_cast<int64_t>(rng());
        ok &= run_case(*ctx, *isort64, in, "i64 signed");
    }

    {
        auto res =
            RadixSort::create(*ctx, SortSpec{.key_bits = 64, .key_order = SortKeyOrder::Float});
        if (!res.ok()) {
            std::fprintf(stderr, "failed to create f64 sort (vulkan %d)\n",
                         static_cast<int>(res.err()));
            return 1;
        }
        auto fsort64 = std::move(res.val());
        auto less64  = [](double a, double b) { return float_order_key(a) < float_order_key(b); };
        auto equal64 = [](double a, double b) { return float_order_key(a) == float_order_key(b); };
        std::vector<double> in(100000);
        for (double& v : in)
            v = static_cast<double>(static_cast<int64_t>(rng() % 2000001) - 1000000) / 1000.0;
        for (double special :
             {-2.5, -0.0, 0.0, std::numeric_limits<double>::infinity(),
              -std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN(),
              std::numeric_limits<double>::denorm_min()})
            in.push_back(special);
        ok &= run_case(*ctx, *fsort64, in, "f64 float", less64, 0, equal64);
    }

    // Key-value: the payload follows its key, and equal keys keep their input order. The values
    // are the original positions, so the reference sort pins the pairing exactly.
    {
        auto res = RadixSort::create(*ctx, SortSpec{.value_bits = 32});
        if (!res.ok()) {
            std::fprintf(stderr, "failed to create key-value sort (vulkan %d)\n",
                         static_cast<int>(res.err()));
            return 1;
        }
        auto pairs = std::move(res.val());

        ok &= check(pairs->spec().value_bits == 32, "pairs spec recorded");

        // The API guard: a spec with values cannot be recorded keys-only, and vice versa.
        AllocationInfo dummy;
        AllocationInfo dummy_scratch;
        ok &= check(!pairs->record(dummy, dummy, dummy_scratch, 1).ok(),
                    "keys-only record rejected for a value spec");
        ok &= check(!sort->record(dummy, dummy, dummy, dummy, dummy_scratch, 1).ok(),
                    "value record rejected for a keys-only spec");

        for (uint32_t n : {4096u, 100000u}) {
            std::vector<uint32_t> keys(n);
            for (uint32_t& v : keys)
                v = static_cast<uint32_t>(rng() % 64u); // many ties: exercises stability
            std::vector<uint32_t> values(n);
            for (uint32_t i = 0; i < n; i++)
                values[i] = i;
            ok &= run_pairs_case(*ctx, *pairs, keys, values, "u32/u32 pairs stable");
        }

        {
            std::vector<uint32_t> keys = random_keys(rng, 100000);
            std::vector<uint32_t> values(keys.size());
            for (uint32_t i = 0; i < values.size(); i++)
                values[i] = i;
            ok &= run_pairs_case(*ctx, *pairs, keys, values, "u32/u32 pairs random");
            ok &= run_pairs_case(*ctx, *pairs, keys, values, "u32/u32 pairs strip", 1u << 14u);
        }
    }

    // 64-bit payloads, and a descending key-value sort.
    {
        auto res = RadixSort::create(*ctx, SortSpec{.value_bits = 64, .descending = true});
        if (!res.ok()) {
            std::fprintf(stderr, "failed to create descending key-value sort (vulkan %d)\n",
                         static_cast<int>(res.err()));
            return 1;
        }
        auto pairs64 = std::move(res.val());

        std::vector<uint32_t> keys(50000);
        for (uint32_t& v : keys)
            v = static_cast<uint32_t>(rng() % 512u);
        std::vector<uint64_t> values(keys.size());
        for (size_t i = 0; i < values.size(); i++)
            values[i] = (static_cast<uint64_t>(i) << 32u) | i;

        ok &= run_pairs_case(*ctx, *pairs64, keys, values, "u32/u64 pairs descending", 0, true);
    }

    // 64-bit keys with a payload: the widest element the sort carries, so the shape is fitted down
    // the furthest here.
    {
        auto res = RadixSort::create(*ctx, SortSpec{.key_bits = 64, .value_bits = 64});
        if (!res.ok()) {
            std::fprintf(stderr, "failed to create u64/u64 sort (vulkan %d)\n",
                         static_cast<int>(res.err()));
            return 1;
        }
        auto wide = std::move(res.val());
        ok &= check(RadixSort::config_fits(ctx->device_info, wide->spec(), wide->shape()),
                    "u64 pairs shape fits");

        std::vector<uint64_t> keys(100000);
        for (uint64_t& v : keys)
            v = rng() % 512u; // many ties: exercises stability
        std::vector<uint64_t> values(keys.size());
        for (size_t i = 0; i < values.size(); i++)
            values[i] = (static_cast<uint64_t>(i) << 32u) | i;

        ok &= run_pairs_case(*ctx, *wide, keys, values, "u64/u64 pairs stable");
        ok &= run_pairs_case(*ctx, *wide, keys, values, "u64/u64 pairs strip", 1u << 13u);
    }

    // Mixed widths: 64-bit keys, 32-bit payload.
    {
        auto res = RadixSort::create(*ctx, SortSpec{.key_bits = 64, .value_bits = 32});
        if (!res.ok()) {
            std::fprintf(stderr, "failed to create u64/u32 sort (vulkan %d)\n",
                         static_cast<int>(res.err()));
            return 1;
        }
        auto mixed                 = std::move(res.val());
        std::vector<uint64_t> keys = random_keys<uint64_t>(rng, 100000);
        std::vector<uint32_t> values(keys.size());
        for (uint32_t i = 0; i < values.size(); i++)
            values[i] = i;
        ok &= run_pairs_case(*ctx, *mixed, keys, values, "u64/u32 pairs random");
    }

    std::printf("\n%s\n", ok ? "all sort tests passed" : "SORT TESTS FAILED");
    return ok ? 0 : 1;
}
