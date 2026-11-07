/* SPDX-License-Identifier: MPL-2.0 */

#include "lookback.slang"

struct ScanParams {
    uint64_t input;
    uint64_t output;
    uint64_t tile_states;
    uint64_t tile_partial;
    uint64_t tile_inclusive;
    uint32_t num_items;
    uint32_t exclusive;
    uint32_t num_states;
    uint32_t value_bits;
};

[[vk::push_constant]]
uniform ScanParams p;

static const uint SCAN_WAVES = 4;
static const uint SCAN_VPT   = 16;

// Adds one look-back window to `prefix`: the sum of `value` over the window, plus the inclusive
// prefix at the lowest lane whose tile is inclusive. Returns true when the window holds an
// inclusive tile.
bool window_step(bool inclusive, uint64_t value, inout uint64_t prefix)
{
    uint64_t psum = wave_excl_scan(value) + value;
    uint     bits = uint(WaveActiveBallot(inclusive).x);
    prefix += (bits != 0u) ? WaveReadLaneAt(psum, firstbitlow(bits)) : wave_reduce(value);
    return bits != 0u;
}

uint64_t decoupled_prefix32(uint tile, uint lane, uint64_t tile_total)
{
    if (lane == 0)
        ((Atomic<uint64_t>*)p.tile_partial)[tile].store(LB_DESC_PARTIAL | uint64_t(uint32_t(tile_total)),
                                                        MemoryOrder::Relaxed);

    uint64_t prefix = 0;
    if (tile > 0) {
        int base = int(tile) - 1;
        while (true) {
            int      idx = base - int(lane);
            uint64_t d   = (idx >= 0) ? 0ull : LB_DESC_OOB;
            while (!WaveActiveAllTrue((d >> 32) != 0ull)) {
                if (idx >= 0)
                    d = ((Atomic<uint64_t>*)p.tile_partial)[idx].load(MemoryOrder::Relaxed);
            }
            uint     st  = uint(d >> 32) & 3u;
            uint64_t val = uint64_t(uint32_t(d));

            if (window_step(st == LB_ST_INCLUSIVE, val, prefix) || base < int(WAVE_SIZE))
                break;
            base -= int(WAVE_SIZE);
        }
    }

    if (lane == 0)
        ((Atomic<uint64_t>*)p.tile_partial)[tile].store(
            LB_DESC_INCLUSIVE | uint64_t(uint32_t(prefix + tile_total)), MemoryOrder::Relaxed);
    return prefix;
}

// Warp-cooperative decoupled look-back, CUB-style: each lane owns one tile in a
// 32-wide window, the window slides back until an inclusive prefix is found. The
// u64 descriptor is value + status in two accesses, with a Release/Acquire pairing.
uint64_t decoupled_prefix64(uint tile, uint lane, uint64_t tile_total)
{
    if (lane == 0) {
        ((Atomic<uint64_t>*)p.tile_partial)[tile].store(tile_total, MemoryOrder::Relaxed);
        ((Atomic<uint>*)p.tile_states)[tile].store(LB_ST_PARTIAL, MemoryOrder::Release);
    }

    uint64_t prefix = 0;
    if (tile > 0) {
        int base = int(tile) - 1;
        while (true) {
            int      idx   = base - int(lane);
            uint     state = (idx >= 0) ? 0u : 3u; // 3 = out-of-window, non-contributing
            while (!WaveActiveAllTrue(state != 0u)) {
                if (idx >= 0)
                    state = ((Atomic<uint>*)p.tile_states)[idx].load(MemoryOrder::Acquire);
            }
            uint64_t val = 0ull;
            if (state == LB_ST_PARTIAL)
                val = ((Atomic<uint64_t>*)p.tile_partial)[idx].load(MemoryOrder::Relaxed);
            else if (state == LB_ST_INCLUSIVE)
                val = ((Atomic<uint64_t>*)p.tile_inclusive)[idx].load(MemoryOrder::Relaxed);

            if (window_step(state == LB_ST_INCLUSIVE, val, prefix) || base < int(WAVE_SIZE))
                break;
            base -= int(WAVE_SIZE);
        }
    }

    if (lane == 0) {
        ((Atomic<uint64_t>*)p.tile_inclusive)[tile].store(prefix + tile_total, MemoryOrder::Relaxed);
        ((Atomic<uint>*)p.tile_states)[tile].store(LB_ST_INCLUSIVE, MemoryOrder::Release);
    }
    return prefix;
}

// Zeroes the tile descriptor array the look-back gates on: the packed word in tile_partial for
// the 32-bit encoding (status 0 = not published), the state in tile_states for the 64-bit one.
// The other array is only read after its publishing status/state, so it needs no prior value.
// Runs before the scan instead of a transfer fill, so the init->scan dependency is
// compute->compute (no transfer-engine transition).
[shader("compute")]
[numthreads(256, 1, 1)]
void init_compute(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x < p.num_states) {
        if (p.value_bits == 64)
            ((uint*)p.tile_states)[tid.x] = 0u;
        else
            ((uint64_t*)p.tile_partial)[tid.x] = 0ull;
    }
}

[shader("compute")]
[numthreads(WAVE_SIZE * SCAN_WAVES, 1, 1)]
void scan32_compute(uint3 gid : SV_GroupID, uint gi : SV_GroupIndex)
{
    const uint C = 4;
    uint       wave = gi / WAVE_SIZE;
    uint       lane = gi % WAVE_SIZE;
    uint       tile = gid.x * SCAN_WAVES + wave;
    uint       row_elems = WAVE_SIZE * C;
    uint64_t   base = uint64_t(tile) * uint64_t(SCAN_VPT * row_elems);
    if (base >= p.num_items)
        return;

    uint64_t e[SCAN_VPT * C];
    uint64_t vbase = base / uint64_t(C) + uint64_t(lane);
    [ForceUnroll]
    for (uint r = 0; r < SCAN_VPT; r++) {
        uint64_t vidx = base + uint64_t(r * row_elems + lane * C);
        if (vidx + C <= p.num_items) {
            uint4 raw = ((uint4*)p.input)[vbase + uint64_t(r * WAVE_SIZE)];
            e[r * C + 0] = raw.x;
            e[r * C + 1] = raw.y;
            e[r * C + 2] = raw.z;
            e[r * C + 3] = raw.w;
        } else {
            [ForceUnroll]
            for (uint k = 0; k < C; k++)
                e[r * C + k] = (vidx + k < p.num_items)
                                   ? uint64_t(((uint32_t*)p.input)[vidx + k])
                                   : 0ull;
        }
    }

    uint64_t local = 0;
    [ForceUnroll]
    for (uint j = 0; j < SCAN_VPT * C; j++)
        local += e[j];
    uint64_t prefix = decoupled_prefix32(tile, lane, wave_reduce(local));

    uint64_t running = prefix;
    [ForceUnroll]
    for (uint r = 0; r < SCAN_VPT; r++) {
        uint64_t lt = 0;
        [ForceUnroll]
        for (uint k = 0; k < C; k++)
            lt += e[r * C + k];
        uint64_t lane_excl = wave_excl_scan(lt);
        uint64_t row_total = wave_reduce(lt);
        uint64_t run = running + lane_excl;
        uint64_t vidx = base + uint64_t(r * row_elems + lane * C);

        uint32_t o[C];
        [ForceUnroll]
        for (uint k = 0; k < C; k++) {
            o[k] = uint32_t(p.exclusive != 0 ? run : run + e[r * C + k]);
            run += e[r * C + k];
        }
        if (vidx + C <= p.num_items)
            ((uint4*)p.output)[vbase + uint64_t(r * WAVE_SIZE)] = uint4(o[0], o[1], o[2], o[3]);
        else
            [ForceUnroll]
            for (uint k = 0; k < C; k++)
                if (vidx + k < p.num_items)
                    ((uint32_t*)p.output)[vidx + k] = o[k];

        running += row_total;
    }
}

[shader("compute")]
[numthreads(WAVE_SIZE * SCAN_WAVES, 1, 1)]
void scan64_compute(uint3 gid : SV_GroupID, uint gi : SV_GroupIndex)
{
    const uint C = 2;
    uint       wave = gi / WAVE_SIZE;
    uint       lane = gi % WAVE_SIZE;
    uint       tile = gid.x * SCAN_WAVES + wave;
    uint       row_elems = WAVE_SIZE * C;
    uint64_t   base = uint64_t(tile) * uint64_t(SCAN_VPT * row_elems);
    if (base >= p.num_items)
        return;

    uint64_t e[SCAN_VPT * C];
    uint64_t vbase = base / uint64_t(C) + uint64_t(lane);
    [ForceUnroll]
    for (uint r = 0; r < SCAN_VPT; r++) {
        uint64_t vidx = base + uint64_t(r * row_elems + lane * C);
        if (vidx + C <= p.num_items) {
            uint64_t2 raw = ((uint64_t2*)p.input)[vbase + uint64_t(r * WAVE_SIZE)];
            e[r * C + 0] = raw.x;
            e[r * C + 1] = raw.y;
        } else {
            [ForceUnroll]
            for (uint k = 0; k < C; k++)
                e[r * C + k] = (vidx + k < p.num_items)
                                   ? ((uint64_t*)p.input)[vidx + k]
                                   : 0ull;
        }
    }

    uint64_t local = 0;
    [ForceUnroll]
    for (uint j = 0; j < SCAN_VPT * C; j++)
        local += e[j];
    uint64_t prefix = decoupled_prefix64(tile, lane, wave_reduce(local));

    uint64_t running = prefix;
    [ForceUnroll]
    for (uint r = 0; r < SCAN_VPT; r++) {
        uint64_t lt = 0;
        [ForceUnroll]
        for (uint k = 0; k < C; k++)
            lt += e[r * C + k];
        uint64_t lane_excl = wave_excl_scan(lt);
        uint64_t row_total = wave_reduce(lt);
        uint64_t run = running + lane_excl;
        uint64_t vidx = base + uint64_t(r * row_elems + lane * C);

        uint64_t o[C];
        [ForceUnroll]
        for (uint k = 0; k < C; k++) {
            o[k] = p.exclusive != 0 ? run : run + e[r * C + k];
            run += e[r * C + k];
        }
        if (vidx + C <= p.num_items)
            ((uint64_t2*)p.output)[vbase + uint64_t(r * WAVE_SIZE)] = uint64_t2(o[0], o[1]);
        else
            [ForceUnroll]
            for (uint k = 0; k < C; k++)
                if (vidx + k < p.num_items)
                    ((uint64_t*)p.output)[vidx + k] = o[k];

        running += row_total;
    }
}
