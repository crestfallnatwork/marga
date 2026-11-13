/* SPDX-License-Identifier: MPL-2.0 */

#include "pictor_core/algorithms/sort.hpp"

#include "sort.shader.h"

#include <algorithm>
#include <array>
#include <string>

namespace pictor {

namespace {
constexpr uint32_t kRadix = 256;

// Workgroup shared memory the bin kernel declares for a shape, in bytes: the `groupshared` arrays
// in sort.slang, summed. The bin kernel is the widest consumer of the four entry points, so they
// all fit whenever it does. The driver may overlap some of these arrays (it does for 1020 B of
// them on sm_89), so this is an upper bound on the allocation.
constexpr uint64_t bin_shared_bytes(uint32_t wave_size, uint32_t waves, uint32_t vpt,
                                    uint32_t key_bits, uint32_t value_bits)
{
    const uint64_t tile = static_cast<uint64_t>(wave_size) * waves * vpt;
    uint64_t bytes      = static_cast<uint64_t>(kRadix) * 4; // g_hist
    bytes += static_cast<uint64_t>(kRadix) * waves * 4;      // g_warp
    bytes += static_cast<uint64_t>(kRadix) * 4 * 3;          // g_tile, g_scan, g_prefix
    bytes += static_cast<uint64_t>(waves) * 4 * 2;           // g_warp_total, g_warp_base
    bytes += tile * (key_bits / 8);                          // g_keys
    bytes += 4;                                              // g_tile_idx
    bytes += static_cast<uint64_t>(kRadix) * 8;              // g_bin_base
    bytes += tile * (value_bits / 8);                        // g_vals
    return bytes;
}

// Shapes measured on the benchmark harness. A device that is not
// listed keeps the shape the shader declares. The entries are the u32-key baseline: a wider key
// takes proportionally fewer items per thread (see scaled_shape).
struct MeasuredShape {
    uint32_t vendor_id;
    uint32_t device_id;
    uint32_t waves;
    uint32_t vpt;
};

constexpr std::array<MeasuredShape, 1> kMeasuredShapes = {{
    {0x10DE, 0x2684, 12, 18}, // GeForce RTX 4090 (AD102), 46180 B of 49152 B
}};

// The tile's staged bytes are what the shape really controls, so the u32 baseline shapes take
// proportionally fewer items per thread when the key widens. The result is still narrowed to the
// device by fit_to_device.
SortShape scaled_shape(SortShape shape, uint32_t key_bits)
{
    const uint32_t width = std::max(key_bits, 1u);
    shape.vpt            = std::max(1u, shape.vpt * 32u / width);
    return shape;
}

// A spec with its implied end bit filled in.
SortSpec normalized(SortSpec spec)
{
    if (spec.end_bit == 0)
        spec.end_bit = spec.key_bits;
    return spec;
}

bool spec_supported(const SortSpec& spec)
{
    if (spec.key_bits != 32 && spec.key_bits != 64)
        return false;
    if (spec.value_bits != 0 && spec.value_bits != 32 && spec.value_bits != 64)
        return false;
    return spec.begin_bit < spec.end_bit && spec.end_bit <= spec.key_bits;
}

uint32_t place_count(const SortSpec& spec)
{
    return (spec.end_bit - spec.begin_bit + kSortDigitBits - 1) / kSortDigitBits;
}

uint32_t last_place_bits(const SortSpec& spec)
{
    return spec.end_bit - spec.begin_bit - ((place_count(spec) - 1) * kSortDigitBits);
}

// The sort's device tuner: the shader declares the shape, the measured-device table overrides it
// for devices we have data for, and the result is scaled by the key width and fitted to the
// device's limits.
SortShape tuned_shape(const DeviceInfo& device, const SortSpec& spec)
{
    for (const MeasuredShape& shape : kMeasuredShapes) {
        if (shape.vendor_id == device.vendorID && shape.device_id == device.deviceID)
            return RadixSort::fit_to_device(
                device, spec, scaled_shape(SortShape{shape.waves, shape.vpt}, spec.key_bits));
    }

    sort_shader::SortTunables declared{};
    return RadixSort::fit_to_device(
        device, spec, scaled_shape(SortShape{declared.waves, declared.vpt}, spec.key_bits));
}
} // namespace

struct RadixSort::Impl {
    std::unique_ptr<sort_shader::HistogramComputeExecutor> exec_hist;
    std::unique_ptr<sort_shader::OffsetComputeExecutor> exec_offset;
    std::unique_ptr<sort_shader::InitComputeExecutor> exec_init;
    std::unique_ptr<sort_shader::BinComputeExecutor> exec_bin;
    SortSpec spec{};
    SortShape shape{};
};

RadixSort::RadixSort() : _impl(std::make_unique<Impl>()) {}

RadixSort::~RadixSort() = default;

Expected<std::unique_ptr<RadixSort>, vk::Result> RadixSort::create(Context& ctx)
{
    return create(ctx, SortSpec{});
}

Expected<std::unique_ptr<RadixSort>, vk::Result> RadixSort::create(Context& ctx,
                                                                   const SortSpec& spec)
{
    return create(ctx, spec, tuned_shape(ctx.device_info, spec));
}

uint64_t RadixSort::shared_bytes(const DeviceInfo& device, const SortSpec& spec,
                                 const SortShape& shape)
{
    return bin_shared_bytes(device.subgroup_size, shape.waves, shape.vpt, spec.key_bits,
                            spec.value_bits);
}

bool RadixSort::config_fits(const DeviceInfo& device, const SortSpec& spec, const SortShape& shape)
{
    // The rank loop's ballot masks cover lanes 0-31 only, so a wider subgroup would mis-rank.
    if (device.subgroup_size > 32)
        return false;
    if (shape.waves == 0 || shape.vpt == 0)
        return false;

    uint64_t threads = static_cast<uint64_t>(device.subgroup_size) * shape.waves;
    if (threads < kSortRadix || threads > device.maxComputeWorkGroupSize[0] ||
        threads > device.maxComputeWorkGroupInvocations) {
        return false;
    }
    return shared_bytes(device, spec, shape) <= device.maxComputeSharedMemorySize;
}

SortShape RadixSort::fit_to_device(const DeviceInfo& device, const SortSpec& spec, SortShape shape)
{
    if (device.subgroup_size == 0)
        return SortShape{std::max(shape.waves, 1u), std::max(shape.vpt, 1u)};

    const uint32_t min_waves = (kSortRadix + device.subgroup_size - 1) / device.subgroup_size;
    uint32_t waves           = std::max(shape.waves, min_waves);
    uint32_t vpt             = std::max(shape.vpt, 1u);

    while (vpt > 1 && !config_fits(device, spec, SortShape{waves, vpt}))
        vpt--;
    while (waves > min_waves && !config_fits(device, spec, SortShape{waves, vpt}))
        waves--;
    return SortShape{waves, vpt};
}

Expected<std::unique_ptr<RadixSort>, vk::Result>
RadixSort::create(Context& ctx, const SortSpec& spec_in, const SortShape& shape)
{
    const SortSpec spec = normalized(spec_in);
    if (!spec_supported(spec) || !config_fits(ctx.device_info, spec, shape))
        return vk::Result::eErrorFormatNotSupported;

    auto sort          = std::unique_ptr<RadixSort>(new RadixSort());
    sort->_ctx         = &ctx;
    sort->_impl->spec  = spec;
    sort->_impl->shape = shape;

    sort_shader::SortTunables tunables{};
    tunables.waves = shape.waves;
    tunables.vpt   = shape.vpt;

    // The module is specialized for the spec: the tunables first, then the spec's digit range and
    // key order. The host sizes every dispatch from the same spec.
    ShaderDefines defines = tunables.defines();
    defines.add("SORT_KEY_ORDER", std::to_string(static_cast<int>(spec.key_order)));
    defines.add("SORT_KEY_BITS", std::to_string(spec.key_bits));
    defines.add("SORT_FIRST_SHIFT", std::to_string(spec.begin_bit));
    defines.add("SORT_LAST_BITS", std::to_string(last_place_bits(spec)));
    defines.add("SORT_DESCENDING", spec.descending ? "1" : "0");
    defines.add("SORT_VALUE_BITS", std::to_string(spec.value_bits));
    const CompileOptions options{defines.view()};

    auto hist = sort_shader::HistogramComputeExecutor::create(ctx, nullptr, options);
    if (!hist.ok())
        return hist.err();
    sort->_impl->exec_hist = std::move(hist.val());

    auto offs = sort_shader::OffsetComputeExecutor::create(ctx, nullptr, options);
    if (!offs.ok())
        return offs.err();
    sort->_impl->exec_offset = std::move(offs.val());

    auto init = sort_shader::InitComputeExecutor::create(ctx, nullptr, options);
    if (!init.ok())
        return init.err();
    sort->_impl->exec_init = std::move(init.val());

    auto bin = sort_shader::BinComputeExecutor::create(ctx, nullptr, options);
    if (!bin.ok())
        return bin.err();
    sort->_impl->exec_bin = std::move(bin.val());

    return sort;
}

const SortSpec& RadixSort::spec() const
{
    return _impl->spec;
}

SortShape RadixSort::shape() const
{
    return _impl->shape;
}

// The tile is the shader's BLOCK * VPT, and BLOCK is the reflected workgroup size rather than a
// re-derived subgroup_size * waves: a shader whose [numthreads] ignored the tunable would then
// fail loudly here instead of silently mis-sorting.
uint32_t RadixSort::tile_size() const
{
    return _impl->exec_bin->wgX * _impl->shape.vpt;
}

uint32_t RadixSort::strip_tiles(uint32_t strip_items) const
{
    uint32_t tile  = tile_size();
    uint32_t tiles = strip_items != 0 ? (strip_items + tile - 1) / tile : kSortDefaultStripTiles;
    return std::clamp(tiles, 1u, kSortDefaultStripTiles);
}

uint32_t RadixSort::num_places() const
{
    return place_count(_impl->spec);
}

// Number of look-back counter rows actually needed: the counters cover one strip, and only the
// longest strip is ever in use. Sizing to the strip capacity instead would clear (and allocate)
// kRadix * kSortDefaultStripTiles words on every sort, regardless of the input length.
uint32_t RadixSort::counter_tiles(uint64_t count, uint32_t strip_items) const
{
    uint32_t tile  = tile_size();
    uint32_t cap   = strip_tiles(strip_items);
    uint64_t items = std::min<uint64_t>(static_cast<uint64_t>(cap) * tile, count);
    return std::min(cap, static_cast<uint32_t>((items + tile - 1) / tile));
}

// The scratch layout, shared by scratch_bytes() and record() so the two cannot drift. Sizes are
// implicit in the next offset.
struct ScratchLayout {
    uint64_t hist_off;
    uint64_t offsets_off;
    uint64_t strip_bins_off;
    uint64_t counters_off;
    uint64_t claim_off;
    uint64_t tmp_keys_off;
    uint64_t tmp_values_off;
    uint64_t size;
};

constexpr uint64_t align_up(uint64_t value, uint64_t align)
{
    return ((value + align - 1) / align) * align;
}

ScratchLayout scratch_layout(uint32_t places, uint64_t strips, uint32_t counters, uint64_t count,
                             uint32_t key_bytes, uint32_t value_bytes)
{
    ScratchLayout l{};
    const uint64_t tables = static_cast<uint64_t>(places) * kRadix * sizeof(uint64_t);

    l.hist_off       = 0;
    l.offsets_off    = l.hist_off + tables;
    l.strip_bins_off = l.offsets_off + tables;
    l.counters_off   = l.strip_bins_off + (strips * kRadix * sizeof(uint64_t));
    l.claim_off    = l.counters_off + (static_cast<uint64_t>(kRadix) * counters * sizeof(uint32_t));
    l.tmp_keys_off = align_up(l.claim_off + sizeof(uint32_t), 8);
    l.tmp_values_off = align_up(l.tmp_keys_off + (count * key_bytes), 8);
    l.size           = l.tmp_values_off + (count * value_bytes);
    return l;
}

uint64_t RadixSort::scratch_bytes(uint64_t count, uint32_t strip_items) const
{
    uint32_t strip  = strip_tiles(strip_items) * tile_size();
    uint64_t strips = count == 0 ? 0 : (count + strip - 1) / strip;
    return scratch_layout(num_places(), strips, counter_tiles(count, strip_items), count,
                          _impl->spec.key_bits / 8, _impl->spec.value_bits / 8)
        .size;
}

Expected<uint32_t, vk::Result> RadixSort::record(const AllocationInfo& keys_in,
                                                 AllocationInfo& keys_out, AllocationInfo& scratch,
                                                 uint64_t count, uint32_t strip_items,
                                                 StreamHandle handle)
{
    if (_impl->spec.value_bits != 0)
        return vk::Result::eErrorFormatNotSupported;
    return record_impl(keys_in, keys_out, nullptr, nullptr, scratch, count, strip_items, handle);
}

Expected<uint32_t, vk::Result> RadixSort::record(const AllocationInfo& keys_in,
                                                 AllocationInfo& keys_out,
                                                 const AllocationInfo& values_in,
                                                 AllocationInfo& values_out,
                                                 AllocationInfo& scratch, uint64_t count,
                                                 uint32_t strip_items, StreamHandle handle)
{
    if (_impl->spec.value_bits == 0)
        return vk::Result::eErrorFormatNotSupported;
    return record_impl(keys_in, keys_out, &values_in, &values_out, scratch, count, strip_items,
                       handle);
}

Expected<uint32_t, vk::Result> RadixSort::record_impl(const AllocationInfo& keys_in,
                                                      AllocationInfo& keys_out,
                                                      const AllocationInfo* values_in,
                                                      AllocationInfo* values_out,
                                                      AllocationInfo& scratch, uint64_t count,
                                                      uint32_t strip_items, StreamHandle handle)
{
    if (count == 0) {
        return 0u;
    }

    uint32_t places   = num_places();
    uint32_t tile     = tile_size();
    uint32_t tiles    = strip_tiles(strip_items);
    uint32_t strip    = tiles * tile;
    uint64_t strips   = (count + strip - 1) / strip;
    uint32_t counters = counter_tiles(count, strip_items);

    const ScratchLayout l = scratch_layout(places, strips, counters, count,
                                           _impl->spec.key_bits / 8, _impl->spec.value_bits / 8);
    if (scratch.size < l.size) {
        return vk::Result::eErrorUnknown;
    }

    AllocationInfo tmp_keys = scratch;
    tmp_keys.offset         = l.tmp_keys_off;
    tmp_keys.size           = count * (_impl->spec.key_bits / 8);
    tmp_keys.address        = scratch.address + l.tmp_keys_off;

    AllocationInfo tmp_values = tmp_keys;
    tmp_values.offset         = l.tmp_values_off;
    tmp_values.size           = count * (_impl->spec.value_bits / 8);
    tmp_values.address        = scratch.address + l.tmp_values_off;

    const uint64_t offsets_addr    = scratch.address + l.offsets_off;
    const uint64_t strip_bins_addr = scratch.address + l.strip_bins_off;

    sort_shader::SortParams params{};
    params.keys_in    = keys_in.address;
    params.hist       = scratch.address + l.hist_off;
    params.offsets    = offsets_addr;
    params.counters   = scratch.address + l.counters_off;
    params.claim      = scratch.address + l.claim_off;
    params.num_items  = count;
    params.num_places = places;

    _ctx->fill(scratch, static_cast<size_t>(l.hist_off),
               static_cast<size_t>(l.offsets_off - l.hist_off), 0u, handle);
    _ctx->barrier(scratch, Access::TransferWrite, Access::ReadWrite, handle);

    // Global histogram: one 2D grid (Vulkan caps a grid dimension at 65535).
    uint64_t hist_tiles = (count + tile - 1) / tile;
    uint64_t gx         = std::min<uint64_t>(hist_tiles, kSortMaxGrid);
    uint64_t gy         = (hist_tiles + gx - 1) / gx;
    if (gy > kSortMaxGrid) {
        return vk::Result::eErrorUnknown;
    }
    params.grid_x = static_cast<uint32_t>(gx);
    _impl->exec_hist->record(static_cast<uint32_t>(gx) * _impl->exec_hist->wgX,
                             static_cast<uint32_t>(gy), 1, params, handle);
    _ctx->barrier(scratch, Access::Write, Access::ReadWrite, handle);

    _impl->exec_offset->record(places * _impl->exec_offset->wgX, 1, 1, params, handle);
    _ctx->barrier(scratch, Access::Write, Access::ReadWrite, handle);

    // The look-back counters carry a phase, so they only need clearing once per sort; the
    // tile-claim counter resets itself inside the bin kernel.
    params.num_states    = kRadix * counters;
    uint32_t init_blocks = (params.num_states + _impl->exec_init->wgX - 1) / _impl->exec_init->wgX;
    _impl->exec_init->record(init_blocks * _impl->exec_init->wgX, 1, 1, params, handle);
    _ctx->barrier(scratch, Access::Write, Access::ReadWrite, handle);

    // The pass buffers alternate; which one a pass writes is pinned to the *last* pass so the
    // result always ends up in the caller's outputs (with the naive parity an odd place count
    // left it in the scratch).
    const bool last_out        = (places - 1) % 2 == 0;
    AllocationInfo prev_keys   = keys_in;
    AllocationInfo prev_values = values_in != nullptr ? *values_in : AllocationInfo{};
    uint32_t phase_n           = 0;
    for (uint32_t pass = 0; pass < places; pass++) {
        const bool to_out  = (pass % 2 == 0) == last_out;
        AllocationInfo dst = to_out ? keys_out : tmp_keys;

        uint64_t const bins_stride = kRadix * sizeof(uint64_t);
        uint64_t const pass_bins   = offsets_addr + (static_cast<uint64_t>(pass) * bins_stride);

        for (uint64_t s = 0; s < strips; s++) {
            uint64_t base    = s * strip;
            uint32_t s_items = static_cast<uint32_t>(std::min<uint64_t>(strip, count - base));
            uint32_t s_tiles = (s_items + tile - 1) / tile;

            params.keys_in     = prev_keys.address;
            params.keys_out    = dst.address;
            params.strip_base  = base;
            params.strip_items = s_items;
            params.num_tiles   = s_tiles;
            params.pass        = pass;
            params.phase       = 1u + (phase_n++ % 3u);
            if (values_in != nullptr) {
                params.values_in  = prev_values.address;
                params.values_out = (to_out ? *values_out : tmp_values).address;
            }

            uint64_t const strip_bins = strip_bins_addr + (s * bins_stride);
            uint64_t const next_bins  = strip_bins_addr + ((s + 1) * bins_stride);
            params.bins_in            = (s == 0) ? pass_bins : strip_bins;
            params.bins_out           = (s + 1 == strips) ? 0ull : next_bins;

            uint32_t bx = std::min(s_tiles, kSortMaxGrid);
            uint32_t by = (s_tiles + bx - 1) / bx;
            _impl->exec_bin->record(bx * _impl->exec_bin->wgX, by, 1, params, handle);
            _ctx->barrier(scratch, Access::Write, Access::ReadWrite, handle);
        }

        _ctx->barrier(keys_out, Access::Write, Access::Read, handle);
        if (values_out != nullptr)
            _ctx->barrier(*values_out, Access::Write, Access::Read, handle);

        prev_keys = dst;
        if (values_in != nullptr)
            prev_values = to_out ? *values_out : tmp_values;
    }

    return places;
}

} // namespace pictor
