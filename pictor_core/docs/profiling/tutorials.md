# Tutorials: worked examples

Worked examples with real captures. The first one is the recommended walkthrough — a program with
two compute kernels and one mesh kernel, traced end to end with no source changes. The others
cover the repo's runnable examples and a calibrated memory-bound workload.

Reference pages: [nsys](nsys.md), [GPU Trace](gputrace.md),
[reading a report](reading-gputrace.md).

## Three kernels: two compute + one mesh

The program runs, in order, as separate submits:

1. uploads (copies + barriers),
2. `add_compute` on a 1M-element buffer,
3. `add_compute` on a second 1M-element buffer,
4. `tile_mesh` (task + mesh) on a 4M-element buffer,

then exits. The essential shape (the runtime emits the labels; no source changes):

```cpp
add->record(kAddCount, 1, 1, add_params);     // -> "add_compute"
ctx->fsync();                                 // one submit
add->record(kAddCount, 1, 1, add_params);     // -> "add_compute"
ctx->fsync();
tile->record(kTileCount, 1, 1, tile_params);  // -> "tile_mesh"
ctx->fsync();
```

### Step 1 — the timeline (nsys)

```bash
nsys profile --trace=vulkan-annotations --vulkan-gpu-workload=individual \
  --stats=true --force-overwrite=true -o three-kernels ./three_kernels
```

The GPU marker summary (captured on an RTX 2070 SUPER, driver 615, nsys 2026.3.2):

```text
 ** Vulkan Range Summary (vulkan_marker_sum):

 Time (%)  Total Time (ns)  Instances  Avg (ns)   ...  Range
     61.9           45,566          4  11,391.5   ...  stream 0
     22.0           16,201          1  16,201.0   ...  tile_mesh
     16.1           11,863          2   5,931.5   ...  add_compute

 ** Vulkan GPU Range Summary (vulkan_gpu_marker_sum):

 Time (%)  Total Time (ns)  Instances  Avg (ns)   ...  Range
     86.3          424,035          1  424,035.0  ...  tile_mesh
     13.7           67,330          2   33,665.0  ...  add_compute
```

What to look at:

- The labels arrive automatically: two `add_compute` rows (one per dispatch) and one `tile_mesh`
  row, plus four `stream 0` submit ranges (the upload submit has no dispatch label).
- The CPU and GPU views differ: the CPU ranges are ~6-16 µs (record/submit cost), while the GPU
  ranges are ~34 µs per compute dispatch and 424 µs for the mesh dispatch. Always use
  `vulkan_gpu_marker_sum` (or the GUI's GPU workload rows) for "which kernel is expensive".
- The mesh kernel dominates here; whether that is expected depends on the workload — the point of
  the capture is that you can now see it.

Open `three-kernels.nsys-rep` in the Nsight Systems GUI to see the per-queue rows with the
labels nested under each submit.

### Step 2 — the counters (GPU Trace)

```bash
sudo scripts/gputrace.sh ./three_kernels \
  --start-after-submits 0 --limit-to-submits 8 \
  --metrics --out gputrace/three-kernels
```

The submit window covers submits `N+1` through `M` (see
[gputrace.md](gputrace.md#the-submit-window)); `0 / 8` captures the whole program. With counters
enabled the layer holds the app at its first submit until the host session is up, so the short
program is traced reliably.

`D3DPERF_EVENTS.xls` from the export (captured on an RTX 4090, driver 595):

```text
event_text	time_ms
stream 0	0.011392
        add_compute	0.011392
stream 0	0.011488
        add_compute	0.011488
stream 0	0.361152
        tile_mesh	0.361152
```

`GPUTRACE_REGIMES.xls` has one metric row per range in the same order
(`stream 0/add_compute`, `stream 0/add_compute`, `stream 0/tile_mesh`).

The window semantics were pinned down with this program (four submits; the upload submit produces
no visible range):

| `--start-after-submits` / `--limit-to-submits` | Captured ranges |
|---|---|
| `0` / `4` | `add_compute`, `add_compute`, `tile_mesh` |
| `0` / `8` | `add_compute`, `add_compute`, `tile_mesh` |
| `1` / `3` | `add_compute`, `add_compute` |
| `1` / `2` | `add_compute` |

Two practical consequences: set `M` to the index of the last submit you want (over-capturing is
harmless), and `N` is how you skip warm-up submits.

What to look at in the GUI:

- **Timeline**: three named ranges; the mesh range is one draw that contains both the Task and
  Mesh stages.
- **Shader Pipelines**: separate `Compute`, `Task` and `Mesh` rows. The `# Warp` column shows the
  theoretical occupancy per stage; the task stage decides how many mesh groups launch, so compare
  the two.
- **Metrics per range**: select a range and read the Summary tab; mesh work shows up in the
  **VTG** warp counter rather than `cs`, and the CTA counter may stay at 0 — do not read
  `ctas_active = 0` as "no work" (see the mesh example below).
- **Tiny ranges are noise**: the percentages are meaningless at these durations; the structure,
  the durations and the instruction counts are the readable part (see
  [reading-gputrace.md](reading-gputrace.md#reading-gputrace_regimesxls)).

## pictor_scan_test: timeline

```bash
nsys profile --trace=vulkan-annotations --vulkan-gpu-workload=individual \
  --stats=true --force-overwrite=true -o scan ./build/algorithms/pictor_scan_test
```

The test runs 12 non-empty sizes for each of {u32, u64} × {exclusive, inclusive}: 48 labeled
submits, 144 labeled dispatches (48 `init_compute`, 24 `scan32_compute`, 24 `scan64_compute`).
What to look at:

- The GPU marker summary: `scan64_compute` dominates GPU time at 58% (avg 48 µs, max 382 µs at
  n=8M), `scan32_compute` 34% (avg 28 µs), `init_compute` 8% (avg 3.2 µs, always small). This is
  the "which dispatch is expensive" answer.
- The CPU marker summary says `stream 0` is 79% of CPU range time and `init_compute` 15%: the CPU
  side is dominated by submit/record overhead and `vkWaitForFences` (44 ms across 97 calls),
  because the test blocks after every case. Do not confuse that with GPU cost.
- SQL: 48 `vkQueueSubmit` workload rows, one per labeled dispatch; the per-label totals
  (scan64 1156.7 µs, scan32 680.1 µs, init 152.7 µs) come straight from `VULKAN_WORKLOAD`.
- Pipeline creation feedback: three pipelines, 1.01 ms / 167 µs / 42 µs creation time (stage
  events 823 µs / 151 µs / 30 µs) — the Slang JIT + driver compile cost you pay once per
  executor.

## pictor_scan_test: GPU Trace

```bash
sudo scripts/gputrace.sh ./build/algorithms/pictor_scan_test \
  --limit-to-submits 8 --metrics --out gputrace/scan
```

The event list for an 8-submit capture:

```text
event_text	time_ms
init_compute	0.003392
scan32_compute	0.005376
stream 0	0.012736
        init_compute	0.00336
        scan32_compute	0.005472
stream 0	0.012192
        init_compute	0.002976
        scan32_compute	0.005536
... (8 submits in total)
```

What you see and what it means:

- Each submit contains two dispatches: `init_compute` (~3 µs) and `scan32_compute` (~5.4 µs);
  the submit range is ~12-13 µs. The sum of the dispatches is ~8.5 µs, so roughly a third of
  each submit range is fixed overhead (copies, barriers, queue work).
- The window covers the first 8 test cases (n = 1..1000), which use only the 32-bit scan path —
  that is why `scan64_compute` never appears. Raise `--limit-to-submits` (or `--max-duration-ms`)
  to reach the big sizes, or run the 150M driver below.
- The metrics are **not meaningful at this scale**. From `GPUTRACE_REGIMES.xls` for a
  representative `scan32_compute` row: sm 0.012%, l1tex 0.008%, lts 0.65%, dramc 0.008%,
  `ctas_active` 0.011%, `warps_active_shader_cs` 0.010%, `warps_inactive_sm_idle` 99.73%,
  752 instructions, L1 hit 100%. Every percentage is effectively zero: two small dispatches
  cannot occupy a 128-SM GPU. Do not read "L2 is higher than SM" as "memory-bound" here — the
  readable facts are the durations, the dispatch structure, the instruction counts (~400 for
  init, ~750 for scan32), and the L1 hit rate.
- What to look at and why: this trace validates the instrumentation (labels, nesting, per-range
  metrics all arrive) and the timing model of the test. For optimization you need a bigger
  workload — see "What a meaningful sample looks like".

## example_compute

Timeline:

```bash
nsys profile --trace=vulkan-annotations --vulkan-gpu-workload=individual \
  --stats=true --force-overwrite=true -o add ./build/examples/compute/example_compute
```

One submit, one labeled dispatch: the workload table has `vkQueueSubmit` (10,721 ns) and
`add_compute` (4,065 ns, `correlationId` 2). The correlation join shows the label begin/end API
calls on the CPU side (270.868 ms and 270.880 ms in trace time) against the GPU range — useful
for seeing how much of the submit is the dispatch.

GPU Trace:

```bash
sudo scripts/gputrace.sh ./build/examples/compute/example_compute \
  --start-after-submits 0 --limit-to-submits 1 --metrics --out gputrace/add
```

A representative regimes row:

```text
add_compute  duration 3.58 us  sm 0.0028%  l1tex 0.0031%  lts 0.28%  dramc 0.086%
             gr_active 49.9%  ctas_active 0.025%  warps_cs 0.025%
             inst 114  alu 16  fma 8  L1 hit 0%
```

What to look at and why:

- The dispatch is 1024 threads = 32 warps = 16 CTAs of 64. With ~114 instructions total, this is
  ~3.5 instructions per warp: the kernel is launch/drain-dominated, and the percentages confirm
  it (nothing is busy). There is nothing to optimize here; this is the plumbing test.
- `dispatch_count` reads 0 despite the one dispatch — a reminder that tiny ranges under-count
  sampled counters.
- To get a useful compute sample from this shader, raise `kCount` or record it in a loop; the
  runtime supports creating the executor once and re-recording.
- The SDK one-shot variant (`--sdk --no-metrics`) traces exactly the scoped block; it is the
  cleanest way to trace one pass of a short-lived app (see
  [gputrace.md](gputrace.md#app-driven-one-shot-capture-sdk)).

## example_mesh

Timeline:

```bash
nsys profile --trace=vulkan-annotations --vulkan-gpu-workload=individual \
  --stats=true --force-overwrite=true -o mesh ./build/examples/mesh/example_mesh
```

The example dispatches twice (via `record`, 7 task groups, then `record_raw`): two `tile_mesh`
labels, CPU ranges 21.8 µs and 1.8 µs, GPU 4.8 µs for the first. One quirk: only one of the two
labels got a `VULKAN_WORKLOAD` row in this capture — the second submit is a 1.3 µs range and the
app exits immediately after it. If you need every GPU range from a short-lived app, keep the app
alive briefly after the last submit, or use the debug-marker table (`VULKAN_DEBUG_API`), which
has both labels.

GPU Trace:

```bash
sudo scripts/gputrace.sh ./build/examples/mesh/example_mesh \
  --start-after-submits 0 --limit-to-submits 2 --metrics --out gputrace/mesh
```

Two `tile_mesh` rows, 2.912 µs and 0.927 µs. Representative metrics:

```text
tile_mesh  duration 2.91 us  sm 0.0049%  lts 0.29%  dramc 0.033%
           gr_active 89.8%  ctas_active 0  warps_vtg 0.014%
           inst 163  alu 72  fma 38  fmaheavy 34  xu 6  L1 hit 11.1%
```

What to look at and why:

- In this capture, task/mesh work shows up in the **VTG** warp counter, not the compute (`cs`)
  counter, and the CTA counter stays at 0 — do not read `ctas_active = 0` as "no work"; the warp
  counter and the GR activity (89.8%) show the work happened.
- In the GUI, the Shader Pipelines view lists **Task** and **Mesh** as separate rows. The `# Warp`
  column gives the theoretical occupancy per stage; the task stage decides how many mesh groups
  launch, so compare the two — a task stage that launches few groups leaves the mesh stage
  underfilled.
- This capture was made without RTS (`Real-Time Shader Profiler: Disabled` in
  `REPRO_INFO.xls`), so there are no sampled stalls. On Ampere+ hardware, rerun with `--rts` for
  per-stage samples (see the caveat in [troubleshooting.md](troubleshooting.md)).
- Like the compute samples, the percentages are meaningless at 1-3 µs; the structure and counts
  are the readable part.

## What a meaningful sample looks like

The repo examples are too small to saturate a modern GPU. To calibrate what "real" numbers look
like, a scratch driver (a single 150M-element u32 exclusive scan built against pictor_core's
`Scan`) was traced the same way:

```bash
sudo scripts/gputrace.sh ./scan-150m \
  --start-after-submits 0 --limit-to-submits 4 --metrics --out gputrace/scan150m
```

```text
init_compute     3.232 us  sm 0.90%   l1tex 1.58%   lts 4.47%   dramc 21.0%   ctas 13.9%
scan32_compute   1285.5 us sm 8.09%   l1tex 15.05%  lts 28.51%  dramc 92.46%  ctas 16.6%
                            dramc read 46.35%  write 46.10%
                            warps_cs 32.1%  inactive_sm_active 67.5%  inactive_sm_idle 0.375%
                            inst 118.8M  alu 50.8M  fma 25.6M  xu 146k
                            L1 hit 72.3%  L2 hit 56.5%
```

How to read it:

- **DRAM at 92% of peak with SM at 8%** — the scan is memory-bandwidth-bound, not compute-bound.
  Cross-check: 150M u32 elements is 600 MB read plus 600 MB written in 1.29 ms ≈ 933 GB/s, about
  92% of the 4090's ~1 TB/s. The two numbers agree, which is why the DRAM row is believable.
- The read and write throughputs are balanced (46% each) — expected for a scan that streams the
  input and writes the output.
- `warps_inactive_sm_active` 67.5% with `warps_cs` 32.1% says resident warps are stalled most of
  the time: this is memory latency under a bandwidth-limited workload, not launch starvation
  (`warps_inactive_sm_idle` is 0.375%).
- 118.8M instructions for 150M elements is plausible (about 0.8 instructions per element across
  the multi-phase scan).
- The `init_compute` phase, by contrast, is a small dispatch and its numbers move between
  captures (dramc 21% in one capture, 62% in the repeat) — a small-range blip, not a diagnosis.
  This is the same noise the repo examples are made of.
- What to look at and why: this is the calibration point. A row where the top throughput is
  above ~50% and the counters are internally consistent is worth acting on; rows in the 0.01%
  range are not.
