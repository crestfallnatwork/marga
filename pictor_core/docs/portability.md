# Device requirements and portability: `scan` and `radix sort`

What the two GPU primitives require from a device and driver, what happens when a requirement is
missing, and what it would take to run on more hardware. Sources are the shader and host sources,
SPIR-V disassembled with the pinned Slang build, the Vulkan/SPIR-V specifications, and measured
vendor behavior. Claims are marked:

- **[code]** verified in the sources or the generated SPIR-V,
- **[spec]** a Vulkan/SPIR-V requirement,
- **[inferred]** vendor behavior or measurement.

Categories:

- **A — won't work at all**: wrong results, a hang, or device/pipeline creation failure, with no
  fallback in-tree.
- **B — works, a little slow**: correct, but degraded relative to the desktop tuning target.
- **C — works, disastrously slow**: correct but pathological; avoid unless the caller chooses it.
- **D — fine everywhere**: satisfied by core Vulkan 1.1+/1.3 on every real driver; listed so the
  reader knows what is *not* a concern.

The largest portability item is not in the kernels: `Context::create` only accepts devices with
mesh-shader support and unconditionally enables several extensions the primitives never use (see
[Runtime gate](#runtime-gate-a9a10)). The second is the decoupled look-back's forward-progress
assumption ([A1](#a1-inter-workgroup-forward-progress)). Details below.

## A. Won't work at all

| | Requirement | Where | If missing | What we can do |
|---|---|---|---|---|
| A1 | Inter-workgroup forward progress | look-back spins in both primitives | hang (spec: no guarantee) | reduce-then-scan or Decoupled Fallback; see [polyfills](#resolutions-and-polyfills) |
| A2 | Subgroup size ≤ 32 | ballot masks in both primitives | silently wrong results | 64-bit masks; or reject >32 (now enforced) |
| A3 | `shaderInt64` | 64-bit addresses, counters, prefixes | no context (device selection rejects) | none — intrinsic to the design |
| A4 | 64-bit buffer atomics | scan descriptors, sort histogram | no device (feature enabled unconditionally) | 32-bit two-word protocols; two-level histogram |
| A5 | Subgroup extended types (64-bit shuffles) | the wave scans in both | no device (feature enabled unconditionally) | a 32-bit scan variant for u32-only devices |
| A6 | Buffer device address | every buffer | no device | none — the executors are BDA-based |
| A7 | `synchronization2` | host barriers between dispatches | no device | mechanical `vkCmdPipelineBarrier` fallback |
| A8 | Vulkan 1.3 instance | `Context::create` | no context | honest floor: subgroups + BDA + 64-bit atomics |
| A9 | Mesh shader (inherited) | device selection only | no context | make the mesh requirement per-consumer |
| A10 | Unused extensions (inherited) | atomic float, memory budget, dynamic rendering, maintenance4, push descriptors | no device | enable per need |
| A11 | ≥ 256 invocations | sort floor; scan `init_compute` | creation failure | two bins per thread; 128-thread init |
| A12 | Ballot/vote/shuffle subgroup ops | both primitives | pipeline creation fails | theoretical only — real drivers expose them |
| A13 | Scan grid ≤ `maxComputeWorkGroupCount` | scan 1D dispatch | > ~5.4e8 elements on min-grid devices | 2D grid like the sort |
| A14 | Scan count arithmetic | `uint32_t` tile math | silent no-op at the top of the range | 64-bit intermediate or reject |
| A15 | Alignment contracts | vector loads, 64-bit keys, scratch | undefined behavior | validate in `record` |
| A16 | Huge single allocations | one buffer per array | allocation fails | split buffers / chunk |

### A1. Inter-workgroup forward progress

Both primitives have workgroups that spin on flags written by *other* workgroups in the same
dispatch: the scan's `decoupled_prefix32`/`decoupled_prefix64` slide a window over predecessor
descriptors, and the sort's `lb_backtrack_32` walks predecessor counters per digit. Vulkan does not
guarantee that a spinning workgroup eventually makes progress, nor that a not-yet-launched
workgroup gets scheduled while another spins [spec: Khronos' memory-model guidance and the open
Vulkan-Docs issue #2233 both state there is no forward-progress guarantee for compute workgroups].
On hardware where that does not hold, the dispatch **hangs** — no timeout, no recompute.

Devices: measured failures on Apple M-series and ARM Mali; NVIDIA/AMD/Intel desktop parts and
recent Intel iGPUs met the weaker "running blocks make progress" property [inferred]. Adreno and
PowerVR are untested. Very old desktop parts never get this far (they fail the [runtime
gate](#runtime-gate-a9a10) first).

One asymmetry: the sort claims tiles with an atomic (`p.claim`), so a tile's predecessors were
claimed earlier and are already running; it only needs running blocks to progress. The scan assigns
tiles statically by workgroup id, so its predecessors may not be scheduled yet on a scheduler that
does not run workgroups roughly in order. The sort is therefore the more robust of the two, but
neither has a fallback.

### A2. Subgroup size > 32

Every ballot mask keeps only the low 32 bits (`WaveActiveBallot(...).x`), and the sort's rank loop
shifts a 32-bit `1` by `lane` [code]. On a 64- or 128-lane subgroup (AMD GCN/Vega, some Adreno
[inferred]) lanes ≥ 32 are absent from the masks: ranks and prefixes are computed from the wrong
population, the scan's window search never finds an inclusive lane ≥ 32, and the shift is undefined
for lane ≥ 32. The output is silently wrong — there is no creation error.

This branch enforces the requirement: `RadixSort::config_fits` rejects `subgroup_size > 32` (so both
the default `create` and an explicit shape fail cleanly) and `Scan::create` rejects it as well.
Polyfill: widen the masks to 64 bits (combine the ballot's `.x`/`.y`, use `firstbitlow64` and
`countbits64`, shift `1ull`); the sort's eight-ballot match loop doubles. Pinning 32 via
`VK_EXT_subgroup_size_control` also works where the device offers it, but devices that cannot pin
would still need the widened path.

### Runtime gate (A9/A10)

`Context::create` selects a device that advertises `VK_EXT_mesh_shader` with both mesh and task
shaders, and enables unconditionally: mesh shaders, `maintenance4`, push descriptors, atomic float,
memory budget, atomic int64, subgroup extended types, BDA, sync2, dynamic rendering and
`shaderInt64`. Scan and sort need **none** of the mesh/float/budget/rendering set and bind no
descriptors at all; they need `shaderInt64`, 64-bit buffer atomics, subgroup extended types, BDA
and sync2. The inherited gate is what keeps the primitives off most mobile hardware — before
64-bit atomics (available on roughly 31% of Android devices [inferred]) even matter.

| Requirement | Enabled by | Needed by scan/sort? | If absent |
|---|---|---|---|
| Vulkan 1.3 instance | `Context` | yes (subgroups, BDA, 64-bit atomics) | no context (A8) |
| `VK_EXT_mesh_shader` + mesh/task features | `Context` | **no** | device selection rejects (A9) |
| `shaderInt64` | `Context` | **yes** | no context (A3) |
| `shaderBufferInt64Atomics` | `Context` | **yes** | no device (A4) |
| `shaderSubgroupExtendedTypes` | `Context` | **yes** | no device (A5) |
| `bufferDeviceAddress` | `Context` | **yes** | no device (A6) |
| `synchronization2` | `Context` | **yes** (host) | no device (A7) |
| `maintenance4`, `dynamicRendering` | `Context` | **no** | no device (A10) |
| `VK_KHR_push_descriptor` | `Context` | **no** (no descriptors bound) | no device (A10) |
| `VK_EXT_shader_atomic_float` | `Context` | **no** | no device (A10) |
| `VK_EXT_memory_budget` | `Context` | **no** | no device (A10) |
| `VK_NV_shader_subgroup_partitioned` | `Context` | **no** (parked WaveMatch removed) | skipped if absent |
| `VK_KHR_vulkan_memory_model` | not enabled | no — modules use GLSL450 | — |
| subgroup size control | not enabled | not currently | pipeline uses the default `subgroupSize` |

## B. Works, a little slow

- **16-lane subgroups (Mali)**: correct, but the sort's minimum shape becomes 16 waves, so `g_warp`
  alone is 16 KiB and the fitted tile is small; the scan runs 64-thread blocks with 1024-element
  tiles [code].
- **The bin-kernel cost model overcounts by 1 KiB**: it includes `g_hist`, which the bin kernel does
  not declare (the compiler drops it per entry point) [code: SPIR-V Workgroup variables]. The model
  is an upper bound by design — the driver also aliases some arrays — so `fit_to_device` can narrow
  or reject a shape the driver could run. At the minimum shape, u64 keys-only is 68 B over the
  16 KiB guarantee *by the model*, not by the hardware.
- **Vector loads are declared `Aligned 4`/`Aligned 8`**: Slang emits the element alignment, so a
  driver may scalarize the scan's 128-bit loads [code: SPIR-V]. Correct either way.
- **Hardcoded grid cap**: the sort clamps to the guaranteed 65535 per dimension instead of the
  device's real `maxComputeWorkGroupCount` (NVIDIA allows 2^31−1); only histogram parallelism at
  very large sizes is affected.
- **64-bit shuffles for 32-bit tile scans**: the sort's tile scan is `uint64_t` even for u32 keys,
  which requires the extended-types feature and is emulated as two shuffles on some hardware.
- **Device-scope atomics on groupshared memory**: correct; workgroup scope would suffice [code:
  SPIR-V scope].
- **Bench timestamps**: the Vulkan harnesses use `GpuTimer`, which checks `timestampValidBits` and
  `timestampPeriod` at creation and falls back to a host clock when timestamps are unsupported, so
  devices with no timestamp support no longer report 0 ms readings.

## C. Works, disastrously slow

- **Low-cardinality keys**: the histogram does one shared `InterlockedAdd` per item per place; with
  all-equal digits the threads serialize on one bin. A per-warp sub-histogram fixes it.
- **Long look-back chains**: if tiles complete in reverse order the walk (and its counter traffic)
  is O(tiles). Measured at 134M: ~2.0 ms of the bin kernel's 5.3 ms; a one-step walk is as fast as
  no walk, so the cost is chain length, not step latency [inferred].
- **Tiny caller-chosen strips**: each strip is a separate dispatch with a host barrier and a 64-bit
  carry. The default (65535 tiles per strip) is fine; a caller setting `strip_items` to a few tiles
  pays the fixed cost per tile.
- **The polyfills themselves**: reduce-then-scan is 1.33–1.49× a single-pass fallback; 64-bit mask
  widening doubles the match loop; two-bins-per-thread costs registers and shared memory.
- **16-lane occupancy**: the sort's `g_warp` is 16 KiB at 16 waves, so fitted shapes land at low
  `vpt` with one or two blocks per SM.

## D. Fine everywhere

Workgroup barriers and memory fences; 32-bit buffer/shared atomics; `countbits`/`firstbitlow`;
indexed shuffles, ballot and vote in compute on every real driver; `shaderInt64` (given the runtime
gate); the scan's 128-thread workgroups and 56-byte push constants; the sort's 128-byte push
constants (exactly the guaranteed `maxPushConstantsSize`); the scan uses no groupshared memory at
all; the sort's grid clamps (2D histogram with an overflow check, ≤ 65535 tiles for the bin
kernel); descriptor-free pipelines (no descriptor-count limits apply); the sort's scratch
alignment; single graphics+compute queue.

## Floors: shared memory and workgroup size

Minimum legal bin-kernel shape (`waves = ceil(256/subgroup_size)`, `vpt = 1`), in bytes:

| subgroup | waves | keys / values | declared `groupshared` | host model |
|---|---|---|---|---|
| 32 | 8 | u32 / – | 14404 | 15428 |
| 32 | 8 | u64 / – | 15428 | 16452 |
| 32 | 8 | u64 / u64 | 17476 | 18500 |
| 16 | 16 | u32 / – | 22660 | 23684 |
| 64 | 4 | u32 / – | 10276 | 11300 |

The scan uses zero groupshared memory [code: SPIR-V]. The Vulkan-guaranteed
`maxComputeSharedMemorySize` is 16384 [spec], so u32 keys-only fits the guarantee while the model
still rejects u64 keys-only by 68 B (see [B](#b-works-a-little-slow)). Real mobile GPUs are usually
32 KB and desktop 48 KB or more, so this only bites 16 KB devices.

Workgroup invocations: the sort needs at least 256 (one digit bin per thread) and the scan's
`init_compute` is 256, while Vulkan guarantees only 128 [spec]. Both fail cleanly on a
strict-minimum device.

## Huge inputs (host-side limits)

| Limit | Behavior |
|---|---|
| Scan count is `uint32_t` | up to 2^32−1 elements; the tile math wraps at the very top of the range (silent no-op) |
| Scan dispatch is 1D | exceeds the guaranteed grid beyond ~5.4e8 u32 / 2.7e8 u64 elements on min-grid devices |
| Sort count is `uint64_t` | 64-bit scratch math and an explicit scratch-size check |
| Sort histogram grid is 2D | graceful failure when it cannot fit the guaranteed grid |
| Buffers are single BDA allocations | `maxBufferSize`/`maxMemoryAllocationSize` ≥ 2^30 [spec]; a 268M-element u64-key sort needs 2 GiB per array |

## Resolutions and polyfills

1. **Forward progress**: *reduce-then-scan* (per-tile sums, a scan of the tile sums, then intra-tile
   scans plus offsets; 3n traffic vs 2n, measured 1.33–1.49× a single-pass fallback [inferred]) or
   the *Decoupled Fallback* (a spinning block recomputes the blocking tile from the still-readable
   input; ≈memcpy on Mali-G78 and 1.006× memcpy on a 2080 Super [inferred]; costs shared memory and
   registers). For the sort, the analogue is a per-strip partial pass plus a scan of the partials
   before each bin pass.
2. **Subgroup > 32**: 64-bit masks as in [A2](#a2-subgroup-size--32), or the rejection now in
   place.
3. **No 64-bit atomics**: the scan can use the two-word protocol it already implements for 64-bit
   values; the sort can keep 32-bit histogram counts while `count < 2^32` (two 32-bit counters per
   bin otherwise).
4. **Fewer than 256 invocations**: two digits per thread in the sort (each thread owns `gi` and
   `gi + 128`; the warp and tile scans process two elements per lane) and a 128-thread init in the
   scan.
5. **16 KB shared-memory devices**: drop the unused `g_hist` from the cost model, shrink `g_warp`,
   or union the rank-time arrays with the staging array.
6. **Huge inputs**: a 2D dispatch and 64-bit intermediates in the scan, plus a buffer-size check.

## Top risks

1. The look-back spins: no spec guarantee, a real hang on Mali/Apple, no fallback in-tree.
2. Subgroup > 32 was silent corruption; now rejected by both primitives.
3. The runtime's device gate is far wider than the primitives' needs and excludes most mobile
   hardware before the kernels' own requirements matter.
4. The 256-invocation/shared-memory floor on minimum-spec devices: graceful, but u64 keys are
   rejected by the model's overcount.
5. Unchecked host-side limits for huge inputs (scan count/grid, buffer sizes).
