/* SPDX-License-Identifier: MPL-2.0 */

// Decoupled look-back (Merrill & Garland) primitives shared by the scan and radix-sort
// kernels. Included textually by reflect.py, so it must not define entry points.
//
// Two counter encodings are used:
//   * 32-bit (radix sort): one word per (tile, digit);
//       bits 31..30 = phase, bit 29 = kind (0 = local, 1 = global), bits 0..28 = value.
//     The phase lets a pass/strip detect counters left by an earlier one without clearing
//     them. Strips cycle through phases 1..3; the cleared state is 0, which is never used, so
//     a never-written counter can never look ready.
//   * 64-bit (scan): status in the high 32 bits, value in the low 32 (see LB_DESC_*).

static const uint LB_VALUE_BITS  = 29u;
static const uint LB_VALUE_MASK  = (1u << LB_VALUE_BITS) - 1u;
static const uint LB_KIND_SHIFT  = 29u;
static const uint LB_KIND_GLOBAL = 1u;
static const uint LB_PHASE_SHIFT = 30u;

uint lb_pack_local(uint phase, uint value)
{
    return (phase << LB_PHASE_SHIFT) | (value & LB_VALUE_MASK);
}

uint lb_pack_global(uint phase, uint value)
{
    return (phase << LB_PHASE_SHIFT) | (LB_KIND_GLOBAL << LB_KIND_SHIFT) | (value & LB_VALUE_MASK);
}

bool lb_ready(uint word, uint phase)
{
    return (word >> LB_PHASE_SHIFT) == phase;
}

bool lb_is_global(uint word)
{
    return ((word >> LB_KIND_SHIFT) & 1u) == LB_KIND_GLOBAL;
}

uint lb_value(uint word)
{
    return word & LB_VALUE_MASK;
}

static const uint64_t LB_DESC_OOB       = 3ull << 32;
static const uint64_t LB_DESC_PARTIAL   = 1ull << 32;
static const uint64_t LB_DESC_INCLUSIVE = 2ull << 32;
static const uint     LB_ST_PARTIAL     = 1u;
static const uint     LB_ST_INCLUSIVE   = 2u;

// Per-thread decoupled look-back over one counter array: walks predecessor tiles until a
// global (inclusive) prefix is found, accumulating the local ones. `digit` selects the column
// (counter slot = tile * stride + digit). Returns the exclusive prefix for `tile`.
uint lb_backtrack_32(uint64_t counters, uint digit, uint stride, uint tile, uint phase)
{
    Atomic<uint>* c    = (Atomic<uint>*)counters;
    uint          excl = 0u;
    int           j    = int(tile) - 1;
    while (j >= 0) {
        uint v;
        do {
            v = c[uint(j) * stride + digit].load(MemoryOrder::Relaxed);
        } while (!lb_ready(v, phase));
        excl += lb_value(v);
        if (lb_is_global(v))
            break;
        j--;
    }
    return excl;
}

// Explicit shuffle-based wave scans. Slang lowers the WavePrefixSum/WaveActiveSum intrinsics
// to OpGroupNonUniformIAdd (scan form), which the NVIDIA Vulkan driver emulates poorly; these
// map to plain OpGroupNonUniformShuffle, as in the fast CUDA reference.
uint64_t wave_excl_scan(uint64_t v)
{
    uint     lane = WaveGetLaneIndex();
    uint64_t incl = v;
    [ForceUnroll]
    for (uint o = 1; o < WAVE_SIZE; o <<= 1) {
        uint     idx = lane >= o ? lane - o : lane;
        uint64_t n   = WaveReadLaneAt(incl, idx);
        if (lane >= o)
            incl += n;
    }
    return incl - v;
}

uint64_t wave_reduce(uint64_t v)
{
    uint lane = WaveGetLaneIndex();
    [ForceUnroll]
    for (uint o = WAVE_SIZE / 2; o > 0; o >>= 1) {
        uint64_t n = WaveReadLaneAt(v, lane ^ o);
        v += n;
    }
    return v;
}
