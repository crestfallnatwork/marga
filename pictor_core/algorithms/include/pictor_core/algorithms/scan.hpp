/* SPDX-License-Identifier: MPL-2.0 */

#pragma once

#include "pictor_core/context/context.hpp"
#include "pictor_core/context/result_type.hpp"
#include "pictor_core/execution/stream.hpp"
#include "pictor_core/memory/allocation_type.hpp"

#include <cstdint>
#include <memory>

namespace pictor {

enum class ScanKind : uint32_t {
    Inclusive = 0,
    Exclusive = 1,
};

// Waves per workgroup in the scan shader; must match SCAN_WAVES in shaders/scan.slang.
inline constexpr uint32_t kScanWaves = 4;

// Elements per thread (VPT * vector width) in the shader; must match SCAN_VPT in
// shaders/scan.slang (4 elements per vector for 32-bit values, 2 for 64-bit).
inline constexpr uint32_t kScanVpt     = 16;
inline constexpr uint32_t kScanItems32 = kScanVpt * 4;
inline constexpr uint32_t kScanItems64 = kScanVpt * 2;

// Single-pass decoupled look-back scan. Owns the compiled pipelines, so create one per context
// and reuse it across calls.
class Scan {
  public:
    static Expected<std::unique_ptr<Scan>, vk::Result> create(Context& ctx);
    ~Scan();

    Scan(const Scan&)            = delete;
    Scan& operator=(const Scan&) = delete;

    // Device bytes required by record() for `count` elements of `value_bits` (32 or 64).
    [[nodiscard]] uint64_t scratch_bytes(uint32_t count, uint32_t value_bits) const;

    // Records the scan into the stream. `scratch` must be at least scratch_bytes(count,
    // value_bits). `input` and `output` hold `count` elements of `value_bits`. Returns the number
    // of tiles dispatched.
    Expected<uint32_t, vk::Result> record(const AllocationInfo& input, AllocationInfo& output,
                                          AllocationInfo& scratch, uint32_t count,
                                          uint32_t value_bits, ScanKind kind,
                                          StreamHandle handle = cDefaultStream);

  private:
    Scan();

    struct Impl;
    std::unique_ptr<Impl> _impl;

    Context* _ctx{nullptr};

    [[nodiscard]] uint32_t tile_size(uint32_t value_bits) const;
};

} // namespace pictor
