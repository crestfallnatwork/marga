/* SPDX-License-Identifier: MPL-2.0 */

// Onesweep LSB radix sort (Adinets & Merrill 2022), 8-bit digits, 32- or 64-bit keys.
//   histogram_compute : global digit counts for every digit place (one pass).
//   offset_compute    : exclusive prefix sum per place -> global bin offsets.
//   init_compute      : clears the look-back counters and the tile-claim counter.
//   bin_compute       : one chained-scan digit-binning pass over one strip.
//
// Large inputs are processed in strips: the look-back counters only ever hold offsets within a
// strip, so a 32-bit counter suffices, and the per-digit bin start of the next strip is carried
// in 64 bits (bins_out -> the next strip's bins_in). Global counts and bin starts are 64-bit.

#include "lookback.slang"

// Workgroup shape, chosen per device by the host. The declared defaults are the conservative
// shape: the host promotes to a larger tile on the devices it has measurements for.
#ifndef PICTOR_WAVES
#define PICTOR_WAVES 8
#endif

#ifndef PICTOR_VPT
#define PICTOR_VPT 16
#endif

// Digit range and key order, injected by the host. These are properties of the compiled module
// rather than device tunables, so they are deliberately not PICTOR_-prefixed (WAVE_SIZE is the
// same kind of thing). The guarded defaults keep the build-time reflection pass self-contained.
#ifndef SORT_KEY_ORDER
#define SORT_KEY_ORDER 0 // 0 = unsigned, 1 = signed (two's complement), 2 = IEEE float
#endif

#ifndef SORT_KEY_BITS
#define SORT_KEY_BITS 32 // 32 or 64
#endif

#ifndef SORT_VALUE_BITS
#define SORT_VALUE_BITS 0 // 0 = keys only, otherwise 32 or 64
#endif

#ifndef SORT_FIRST_SHIFT
#define SORT_FIRST_SHIFT 0
#endif

#ifndef SORT_LAST_BITS
#define SORT_LAST_BITS 8 // bit count of the last digit place, 1..8
#endif

#ifndef SORT_DESCENDING
#define SORT_DESCENDING 0
#endif

// The push-constant block: 12 x uint64_t + 7 x uint32_t = 124 B, tail-padded to 128 B, exactly
// the minimum maxPushConstantsSize Vulkan guarantees.
struct SortParams {
    uint64_t keys_in;
    uint64_t keys_out;
    uint64_t values_in;
    uint64_t values_out;
    uint64_t hist;
    uint64_t offsets;
    uint64_t counters;
    uint64_t claim;
    uint64_t bins_in;
    uint64_t bins_out;
    uint64_t strip_base;
    uint64_t num_items;
    uint32_t strip_items;
    uint32_t num_places;
    uint32_t num_tiles;
    uint32_t num_states;
    uint32_t grid_x;
    uint32_t pass;
    uint32_t phase;
};

[[vk::push_constant]]
uniform SortParams p;

static const uint DIGIT_BITS = 8;
static const uint RADIX      = 256;
static const uint WAVES      = PICTOR_WAVES;
static const uint BLOCK      = WAVE_SIZE * WAVES;
static const uint VPT        = PICTOR_VPT;
static const uint TILE       = BLOCK * VPT;

#if SORT_KEY_BITS == 64
typedef uint64_t sort_key_t;
#else
typedef uint sort_key_t;
#endif

groupshared uint       g_hist[RADIX];
groupshared uint       g_warp[RADIX * WAVES];
groupshared uint       g_tile[RADIX];
groupshared uint       g_scan[RADIX];
groupshared uint       g_prefix[RADIX];
groupshared uint       g_warp_total[WAVES];
groupshared uint       g_warp_base[WAVES];
groupshared sort_key_t g_keys[TILE];
#if SORT_VALUE_BITS == 32
groupshared uint     g_vals[TILE];
#elif SORT_VALUE_BITS == 64
groupshared uint64_t g_vals[TILE];
#endif
groupshared uint     g_tile_idx;
groupshared uint64_t g_bin_base[RADIX];

// The sort key maps the raw key onto an unsigned value whose ascending order is the order the
// caller asked for: two's-complement keys need their sign bit flipped, IEEE floats the usual
// sign-magnitude transform.
sort_key_t sort_key_of(sort_key_t key)
{
#if SORT_KEY_ORDER == 1
    return key ^ (sort_key_t(1) << (SORT_KEY_BITS - 1));
#elif SORT_KEY_ORDER == 2
    return (key & (sort_key_t(1) << (SORT_KEY_BITS - 1))) != 0
               ? ~key
               : (key | (sort_key_t(1) << (SORT_KEY_BITS - 1)));
#else
    return key;
#endif
}

uint digit_of(sort_key_t key, uint shift, uint bits)
{
    sort_key_t k = sort_key_of(key);
#if SORT_DESCENDING
    k = ~k;
#endif
    return uint((k >> shift) & ((sort_key_t(1) << bits) - sort_key_t(1)));
}

uint place_shift(uint place)
{
    return uint(SORT_FIRST_SHIFT) + place * DIGIT_BITS;
}

uint place_bits(uint place, uint num_places)
{
    return place + 1u == num_places ? uint(SORT_LAST_BITS) : DIGIT_BITS;
}

// A key that sorts after every real key in the requested order and range, used to fill the tail of
// a partial tile. It must produce digit `RADIX - 1` for every place, ascending or descending, so
// that the filler occupies the tile's last bin - slots which are never stored, and which no later
// tile's bin base consumes, because a strip is always a whole number of tiles.
sort_key_t pad_key()
{
#if SORT_DESCENDING
    #if SORT_KEY_ORDER == 2
        return ~sort_key_t(0);
    #elif SORT_KEY_ORDER == 1
        return sort_key_t(1) << (SORT_KEY_BITS - 1);
    #else
        return sort_key_t(0);
    #endif
#else
    #if SORT_KEY_ORDER == 0
        return ~sort_key_t(0);
    #else
        return (~sort_key_t(0)) >> 1;
    #endif
#endif
}

[shader("compute")]
[numthreads(WAVE_SIZE * PICTOR_WAVES, 1, 1)]
void init_compute(uint gid : SV_GroupID, uint gi : SV_GroupIndex)
{
    uint idx = gid * BLOCK + gi;
    if (idx < p.num_states)
        ((uint*)p.counters)[idx] = 0u;
    if (idx == 0u)
        ((uint*)p.claim)[0] = 0u;
}

[shader("compute")]
[numthreads(WAVE_SIZE * PICTOR_WAVES, 1, 1)]
void histogram_compute(uint3 gid : SV_GroupID, uint gi : SV_GroupIndex)
{
    uint     tile = gid.x + gid.y * p.grid_x;
    uint64_t base = uint64_t(tile) * uint64_t(TILE);
    if (base >= p.num_items)
        return;

    for (uint place = 0; place < p.num_places; place++) {
        if (gi < RADIX)
            g_hist[gi] = 0u;
        GroupMemoryBarrierWithGroupSync();

        uint shift = place_shift(place);
        uint bits  = place_bits(place, p.num_places);
        [ForceUnroll]
        for (uint i = 0; i < VPT; i++) {
            uint idx = gi + i * BLOCK;
            if (base + uint64_t(idx) < p.num_items) {
                sort_key_t key = ((sort_key_t*)p.keys_in)[base + uint64_t(idx)];
                uint old;
                InterlockedAdd(g_hist[digit_of(key, shift, bits)], 1u, old);
            }
        }
        GroupMemoryBarrierWithGroupSync();

        if (gi < RADIX) {
            uint64_t old;
            InterlockedAdd(((uint64_t*)p.hist)[place * RADIX + gi], uint64_t(g_hist[gi]), old);
        }
        GroupMemoryBarrierWithGroupSync();
    }
}

[shader("compute")]
[numthreads(WAVE_SIZE * PICTOR_WAVES, 1, 1)]
void offset_compute(uint gid : SV_GroupID, uint gi : SV_GroupIndex)
{
    if (gi >= RADIX)
        return;
    uint64_t sum = 0ull;
    for (uint i = 0; i < gi; i++)
        sum += ((uint64_t*)p.hist)[gid * RADIX + i];
    ((uint64_t*)p.offsets)[gid * RADIX + gi] = sum;
}

[shader("compute")]
[numthreads(WAVE_SIZE * PICTOR_WAVES, 1, 1)]
void bin_compute(uint gi : SV_GroupIndex)
{
    if (gi == 0u) {
        uint old;
        InterlockedAdd(((uint*)p.claim)[0], 1u, old);
        g_tile_idx = old;
        if (old == p.num_tiles - 1u)
            ((uint*)p.claim)[0] = 0u;
    }
    GroupMemoryBarrierWithGroupSync();

    uint tile = g_tile_idx;
    if (tile >= p.num_tiles)
        return;

    uint warp   = gi / WAVE_SIZE;
    uint lane   = gi % WAVE_SIZE;
    uint shift  = place_shift(p.pass);
    uint bits   = place_bits(p.pass, p.num_places);
    uint inside = warp * (WAVE_SIZE * VPT);
    uint phase  = p.phase;

    for (uint i = gi; i < RADIX * WAVES; i += BLOCK)
        g_warp[i] = 0u;
    GroupMemoryBarrierWithGroupSync();

    sort_key_t keys[VPT];
    uint       ranks[VPT];

    uint64_t tile_off   = uint64_t(tile) * uint64_t(TILE);
    uint64_t tile_base  = p.strip_base + tile_off;
    uint64_t strip_span = uint64_t(p.strip_items);
    uint64_t tile_items = strip_span - tile_off;
    bool     full_tile  = tile_items >= uint64_t(TILE);

    // Load every key before ranking any of them: the loads are independent, so issuing them
    // together gives the memory system VPT requests in flight per thread instead of one.
    // Partial tiles pad with a key that sorts after every real key (see pad_key).
    if (full_tile) {
        [ForceUnroll] for (uint j = 0; j < VPT; j++)
            keys[j] = ((sort_key_t*)p.keys_in)[tile_base + uint64_t(inside + j * WAVE_SIZE + lane)];
    } else {
        [ForceUnroll] for (uint j = 0; j < VPT; j++)
        {
            uint idx     = inside + j * WAVE_SIZE + lane;
            bool valid   = uint64_t(idx) < tile_items;
            sort_key_t k = ((sort_key_t*)p.keys_in)[tile_base + uint64_t(valid ? idx : 0u)];
            keys[j]      = valid ? k : pad_key();
        }
    }

    [ForceUnroll] for (uint j = 0; j < VPT; j++)
    {
        uint d = digit_of(keys[j], shift, bits);

        // The ballot masks cover lanes 0-31; the host rejects wider subgroups (config_fits).
        uint match = 0xFFFFFFFFu;
        [ForceUnroll] for (uint b = 0; b < DIGIT_BITS; b++)
        {
            uint bit  = (d >> b) & 1u;
            uint vote = uint(WaveActiveBallot(bit != 0u).x);
            match &= (bit != 0u) ? vote : ~vote;
        }
        uint pop  = countbits(match);
        uint lead = firstbitlow(match);
        uint prev = 0u;
        if (lane == lead)
            InterlockedAdd(g_warp[warp * RADIX + d], pop, prev);
        prev       = WaveReadLaneAt(prev, lead);
        uint below = countbits(match & ((1u << lane) - 1u));
        ranks[j]   = prev + below;
    }
    GroupMemoryBarrierWithGroupSync();

    if (gi < RADIX) {
        uint run = 0u;
        [ForceUnroll] for (uint w = 0; w < WAVES; w++)
        {
            uint c                 = g_warp[w * RADIX + gi];
            g_warp[w * RADIX + gi] = run;
            run += c;
        }
        g_tile[gi] = run;
    }

    Atomic<uint>* counter = (Atomic<uint>*)p.counters;
    uint          slot    = tile * RADIX + gi;

    // Publish the tile's own counts before the walk, and before the tile scan: a successor's walk
    // stops at the first tile whose counters are ready, so the earlier this lands the shorter
    // every later chain is.
    if (gi < RADIX)
        counter[slot].store(lb_pack_local(phase, g_tile[gi]), MemoryOrder::Relaxed);

    // exclusive scan over the tile's per-digit counts (warp scan + warp-total scan)
    uint total = 0u;
    if (gi < RADIX) {
        uint v     = g_tile[gi];
        uint in_w  = uint(wave_excl_scan(uint64_t(v)));
        total      = uint(wave_reduce(uint64_t(v)));
        g_scan[gi] = in_w;
    }
    // Every warp publishes its total, including warps that own no digit when WAVES > RADIX/WAVE_SIZE.
    if (lane == 0u)
        g_warp_total[warp] = total;
    GroupMemoryBarrierWithGroupSync();

    if (warp == 0u) {
        uint t  = (lane < WAVES) ? g_warp_total[lane] : 0u;
        uint ex = uint(wave_excl_scan(uint64_t(t)));
        if (lane < WAVES)
            g_warp_base[lane] = ex;
    }
    GroupMemoryBarrierWithGroupSync();

    if (gi < RADIX)
        g_prefix[gi] = g_warp_base[warp] + g_scan[gi];
    GroupMemoryBarrierWithGroupSync();

    // Stage the tile at its in-tile position before the walk instead of after it: the staging
    // stores are then in flight while the walk's loads are outstanding, so the walk's latency is
    // not added to the staging time. The global store still waits for both.
    [ForceUnroll] for (uint j = 0; j < VPT; j++)
    {
        uint d     = digit_of(keys[j], shift, bits);
        uint stage = g_prefix[d] + g_warp[warp * RADIX + d] + ranks[j];
        g_keys[stage] = keys[j];
        // The value has to be staged with its key: the tile is permuted, so a staging slot no
        // longer knows which input index its key came from.
#if SORT_VALUE_BITS != 0
        uint vindex = inside + j * WAVE_SIZE + lane;
        uint src    = uint64_t(vindex) < tile_items ? vindex : uint32_t(tile_items - 1u);
#if SORT_VALUE_BITS == 32
        g_vals[stage] = ((uint*)p.values_in)[tile_base + uint64_t(src)];
#else
        g_vals[stage] = ((uint64_t*)p.values_in)[tile_base + uint64_t(src)];
#endif
#endif
    }

    if (gi < RADIX) {
        uint excl = lb_backtrack_32(p.counters, gi, RADIX, tile, phase);
        counter[slot].store(lb_pack_global(phase, excl + g_tile[gi]), MemoryOrder::Relaxed);
        g_bin_base[gi] = ((uint64_t*)p.bins_in)[gi] + uint64_t(excl) - uint64_t(g_prefix[gi]);
        if (p.bins_out != 0ull && tile == p.num_tiles - 1u)
            ((uint64_t*)p.bins_out)[gi] =
                ((uint64_t*)p.bins_in)[gi] + uint64_t(excl) + uint64_t(g_tile[gi]);
    }
    GroupMemoryBarrierWithGroupSync();

    [ForceUnroll] for (uint u = 0; u < VPT; u++)
    {
        uint idx = gi + u * BLOCK;
        if (full_tile || uint64_t(idx) < tile_items) {
            sort_key_t key = g_keys[idx];
            uint64_t dst = uint64_t(idx) + g_bin_base[digit_of(key, shift, bits)];
            ((sort_key_t*)p.keys_out)[dst] = key;
#if SORT_VALUE_BITS == 32
            ((uint*)p.values_out)[dst] = g_vals[idx];
#elif SORT_VALUE_BITS == 64
            ((uint64_t*)p.values_out)[dst] = g_vals[idx];
#endif
        }
    }
}
