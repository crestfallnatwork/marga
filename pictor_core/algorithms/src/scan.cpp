/* SPDX-License-Identifier: MPL-2.0 */

#include "pictor_core/algorithms/scan.hpp"

#include "scan.shader.h"

namespace pictor {

struct Scan::Impl {
    std::unique_ptr<scan_shader::Scan32ComputeExecutor> exec32;
    std::unique_ptr<scan_shader::Scan64ComputeExecutor> exec64;
    std::unique_ptr<scan_shader::InitComputeExecutor> exec_init;
};

Scan::Scan() : _impl(std::make_unique<Impl>()) {}

Scan::~Scan() = default;

Expected<std::unique_ptr<Scan>, vk::Result> Scan::create(Context& ctx)
{
    // The look-back window ballot reads only lanes 0-31, so a wider subgroup would drop tiles
    // from the window.
    if (ctx.subgroup_size > 32)
        return vk::Result::eErrorFormatNotSupported;

    auto scan  = std::unique_ptr<Scan>(new Scan());
    scan->_ctx = &ctx;

    auto e32 = scan_shader::Scan32ComputeExecutor::create(ctx);
    if (!e32.ok())
        return e32.err();
    scan->_impl->exec32 = std::move(e32.val());

    auto e64 = scan_shader::Scan64ComputeExecutor::create(ctx);
    if (!e64.ok())
        return e64.err();
    scan->_impl->exec64 = std::move(e64.val());

    auto ei = scan_shader::InitComputeExecutor::create(ctx);
    if (!ei.ok())
        return ei.err();
    scan->_impl->exec_init = std::move(ei.val());

    return scan;
}

uint32_t Scan::tile_size(uint32_t value_bits) const
{
    uint32_t items = (value_bits == 64) ? kScanItems64 : kScanItems32;
    return items * _ctx->subgroup_size;
}

uint64_t Scan::scratch_bytes(uint32_t count, uint32_t value_bits) const
{
    uint32_t tile      = tile_size(value_bits);
    uint32_t num_tiles = (count + tile - 1) / tile;
    return static_cast<uint64_t>(num_tiles) * ((2 * sizeof(uint64_t)) + sizeof(uint32_t));
}

Expected<uint32_t, vk::Result> Scan::record(const AllocationInfo& input, AllocationInfo& output,
                                            AllocationInfo& scratch, uint32_t count,
                                            uint32_t value_bits, ScanKind kind, StreamHandle handle)
{
    if (value_bits != 32 && value_bits != 64) {
        return vk::Result::eErrorFormatNotSupported;
    }

    uint32_t tile      = tile_size(value_bits);
    uint32_t num_tiles = (count + tile - 1) / tile;
    if (num_tiles == 0) {
        return 0u;
    }

    uint64_t descriptor_bytes = static_cast<uint64_t>(num_tiles) * sizeof(uint64_t);
    uint64_t states_offset    = 2 * descriptor_bytes;
    uint64_t states_bytes     = static_cast<uint64_t>(num_tiles) * sizeof(uint32_t);
    if (scratch.size < states_offset + states_bytes) {
        return vk::Result::eErrorUnknown;
    }

    scan_shader::ScanParams params{};
    params.input          = input.address;
    params.output         = output.address;
    params.tile_partial   = scratch.address;
    params.tile_inclusive = scratch.address + descriptor_bytes;
    params.tile_states    = scratch.address + states_offset;
    params.num_items      = count;
    params.exclusive      = static_cast<uint32_t>(kind);
    params.num_states     = num_tiles;
    params.value_bits     = value_bits;

    _impl->exec_init->record(num_tiles, 1, 1, params, handle);
    _ctx->barrier(scratch, Access::Write, Access::ReadWrite, handle);

    uint32_t blocks = (num_tiles + kScanWaves - 1) / kScanWaves;
    if (value_bits == 64)
        _impl->exec64->record(blocks * _impl->exec64->wgX, 1, 1, params, handle);
    else
        _impl->exec32->record(blocks * _impl->exec32->wgX, 1, 1, params, handle);

    _ctx->barrier(scratch, Access::Write, Access::ReadWrite, handle);
    return num_tiles;
}

} // namespace pictor
