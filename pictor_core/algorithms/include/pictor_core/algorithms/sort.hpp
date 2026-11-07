/* SPDX-License-Identifier: MPL-2.0 */

#pragma once

#include "pictor_core/context/context.hpp"
#include "pictor_core/context/result_type.hpp"
#include "pictor_core/execution/stream.hpp"
#include "pictor_core/memory/allocation_type.hpp"

#include <cstdint>
#include <memory>

namespace pictor {

// Onesweep LSB radix sort (Adinets & Merrill 2022). 8-bit digit places; 32- or 64-bit keys.
inline constexpr uint32_t kSortDigitBits = 8;
inline constexpr uint32_t kSortRadix     = 1u << kSortDigitBits;

// Largest grid dimension Vulkan guarantees (maxComputeWorkGroupCount).
inline constexpr uint32_t kSortMaxGrid = 65535;

// Default strip length in tiles; the look-back counters cover one strip.
inline constexpr uint32_t kSortDefaultStripTiles = kSortMaxGrid;

// The bin kernel's workgroup shape: `waves` subgroups per block, `vpt` keys per thread. Zero fields
// mean "pick for the device".
struct SortShape {
    uint32_t waves{};
    uint32_t vpt{};
};

// How keys compare: raw bits, two's complement, or IEEE float.
enum class SortKeyOrder {
    Unsigned,
    Signed,
    Float
};

// What an instance sorts and in what order; chosen at create().
struct SortSpec {
    // Key width: 32 or 64. Keys and payloads of 64-bit width must be 8-byte aligned.
    uint32_t key_bits{32};
    // The sorted range `[begin_bit, end_bit)` within the keys.
    uint32_t begin_bit{0};
    uint32_t end_bit{0}; // 0 = key_bits
    // 0 = keys only, otherwise 32 or 64: the width of the payload moved with each key.
    uint32_t value_bits{0};
    SortKeyOrder key_order{SortKeyOrder::Unsigned};
    bool descending{false};
};

// Single-pass-per-digit radix sort; owns the compiled pipelines.
class RadixSort {
  public:
    // u32 keys, ascending, over the whole key.
    static Expected<std::unique_ptr<RadixSort>, vk::Result> create(Context& ctx);

    // Creates the sort for `spec`, picking the shape for this device.
    static Expected<std::unique_ptr<RadixSort>, vk::Result> create(Context& ctx,
                                                                   const SortSpec& spec);

    // Creates the sort with an explicit shape, which must fit the device for this spec (see
    // `config_fits`).
    static Expected<std::unique_ptr<RadixSort>, vk::Result>
    create(Context& ctx, const SortSpec& spec, const SortShape& shape);

    // Workgroup shared memory the bin kernel declares for `spec` and `shape`, in bytes; an upper
    // bound on what the driver allocates.
    [[nodiscard]] static uint64_t shared_bytes(const DeviceInfo& device, const SortSpec& spec,
                                               const SortShape& shape);

    // True when the device can run `spec` and `shape`: subgroup size <= 32, at least one digit bin
    // per thread, workgroup size within the device limits, and `shared_bytes` within
    // `maxComputeSharedMemorySize`.
    [[nodiscard]] static bool config_fits(const DeviceInfo& device, const SortSpec& spec,
                                          const SortShape& shape);

    // Narrows `shape` until the device can run it: `vpt` first, then `waves`, never below one
    // digit bin per thread.
    [[nodiscard]] static SortShape fit_to_device(const DeviceInfo& device, const SortSpec& spec,
                                                 SortShape shape);

    ~RadixSort();

    RadixSort(const RadixSort&)            = delete;
    RadixSort& operator=(const RadixSort&) = delete;

    // Device bytes required by record() for `count` elements. `strip_items` overrides the strip
    // length (0 = default); it must match the value passed to record().
    [[nodiscard]] uint64_t scratch_bytes(uint64_t count, uint32_t strip_items = 0) const;

    // Sorts `count` keys from `keys_in` into `keys_out` in the spec's order. `scratch` must be at
    // least scratch_bytes(count). `strip_items` overrides the strip length (0 = default) and must
    // match `scratch_bytes`; `keys_out` must not overlap `keys_in`. Returns the pass count.
    Expected<uint32_t, vk::Result> record(const AllocationInfo& keys_in, AllocationInfo& keys_out,
                                          AllocationInfo& scratch, uint64_t count,
                                          uint32_t strip_items = 0,
                                          StreamHandle handle  = cDefaultStream);

    // Key-value variant for a spec with `value_bits` set: the payload follows its key and equal
    // keys keep their input order. No output may overlap an input.
    Expected<uint32_t, vk::Result> record(const AllocationInfo& keys_in, AllocationInfo& keys_out,
                                          const AllocationInfo& values_in,
                                          AllocationInfo& values_out, AllocationInfo& scratch,
                                          uint64_t count, uint32_t strip_items = 0,
                                          StreamHandle handle = cDefaultStream);

    // The compiled spec.
    [[nodiscard]] const SortSpec& spec() const;

    // The compiled shape.
    [[nodiscard]] SortShape shape() const;

  private:
    RadixSort();

    struct Impl;
    std::unique_ptr<Impl> _impl;

    Context* _ctx{nullptr};

    [[nodiscard]] uint32_t tile_size() const;
    [[nodiscard]] uint32_t strip_tiles(uint32_t strip_items) const;
    [[nodiscard]] uint32_t counter_tiles(uint64_t count, uint32_t strip_items) const;
    [[nodiscard]] uint32_t num_places() const;

    Expected<uint32_t, vk::Result>
    record_impl(const AllocationInfo& keys_in, AllocationInfo& keys_out,
                const AllocationInfo* values_in, AllocationInfo* values_out,
                AllocationInfo& scratch, uint64_t count, uint32_t strip_items, StreamHandle handle);
};

} // namespace pictor
