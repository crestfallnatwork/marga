# Reading a GPU Trace report

A GPU Trace report is a timeline plus a set of tables; the skill is knowing which row answers
which question. This page is the interpretation reference — how to capture a report is in
[gputrace.md](gputrace.md), and worked examples are in [tutorials.md](tutorials.md).

## The window

The GPU Trace window has two main tabs — **Timeline** and **Shader Source** — and a set of
sections: the trace toolbar, the timeline rows, metric graphs, the Event List, Event Details, the
Instruction Mix, the Real-Time Shader Profiler tabs, and the Information tabs (Summary, Metrics,
Shader Warp Occupancy, Trace Information). NVIDIA's
[GPU Trace UI reference](https://docs.nvidia.com/nsight-graphics/UserGuide/gpu-trace-ui.html)
describes each one.

Start with the **Trace Information** tab: it names the GPU, driver, metric set, clock lock and
warnings. If the metric set is "Async Compute Triage" and clocks are "Locked to Base", the numbers
are triage-grade and comparable between runs, but not peak numbers.

## Timeline rows

- **Context row** — which API context was active. The GR engine executes one context at a time;
  context switches cost time. A healthy headless compute trace has one context.
- **Per-queue rows** — one row per queue, with actions, user markers and synchronization objects.
  Our `stream N` queue labels and the per-dispatch labels (`scan32_compute`, `add_compute`,
  `tile_mesh`) show up here; this is the fastest way to find a pass.
- **Actions row** — draws and dispatches, with the incremental time cost of each successive API
  call.
- **Queue synchronization objects** — wait/signal bars between queues; relevant once pictor uses
  multiple queues.
- **Resource barriers** — captured as events; the "Overlay Barriers" toggle shows how they carve
  up the metric graphs. Use it to see a barrier stall the front end.
- **Subchannel switch overlay** — on Ampere/Ada, switching between work types (draw ↔ dispatch)
  on one queue may insert an implicit front-end barrier. Compute-only workloads rarely hit it,
  but a copy-then-dispatch sequence can.
- **Metrics Data Rows** — the per-unit counters. Hovering shows a tooltip with the counters for
  the point under the cursor or the average over the selected range; regions without samples are
  crossed off. Rows can be pinned or hidden.
- **SM Warp Occupancy** and **Per Shader Warp Occupancy** rows — warp-slot residency over time,
  first by shader type, then per shader (when per-shader data is available).
- **Compute row** — precise compute start/stop from hardware events; Blackwell+ only, not on
  Turing/Ada.

## Metric graphs and unit throughput

The **Unit Throughputs** row overlays the %-of-max throughput of every hardware unit; several
units can be near 100% at the same moment. For compute workloads the ones that matter (the
[NVIDIA System Architecture Guide](https://docs.nvidia.com/nsight-graphics/UserGuide/gpu-trace-system-architecture.html)
describes every unit):

| Row | What it measures | Read it as |
|---|---|---|
| SM | Streaming multiprocessor issue/shader execution | Compute/issue pressure |
| L1TEX | L1 data cache + LSU (loads/stores) + TEX | Memory-instruction pressure at L1 |
| L2 | The shared L2 cache | Working set / L2 bandwidth |
| VRAM | DRAM bandwidth (half duplex: read + write sum) | True memory-bandwidth pressure |
| PCIe | Host↔device traffic (full duplex, overlay) | Upload/download cost |

The world/screen-pipe units belong to the raster pipeline; for pure compute dispatches they stay
at (or near) 0. If one of them is clearly active in a compute-only app, something else is running
on the GPU (GPU Trace collects all GPU activity — check for other processes):

| Row | What it measures |
|---|---|
| PD | Primitive distributor: fetches indices, sends triangles to the vertex shader |
| VAF | Vertex attribute fetch: reads attribute values for the vertex shader |
| PES+VPC | Primitive engine: stream-out and clip/cull across the world-pipe stages |
| RASTER | Rasterizer: turns primitives into fragments and coverage samples |
| PROP | Pre-ROP: API ordering of pixel shading, early/late-Z and blending |
| ZROP / CROP | Depth/stencil updates (ZROP) and color blend / render-target updates (CROP) |

Reading method (the P3 / "peak performance percentage" idea the NVIDIA docs recommend):

1. Select the tightest range that represents the work (a dispatch label, or a hot loop).
2. Find the **highest** unit throughput in the selection.
   - SM near 100% → compute/issue-bound; drill into SM Throughput (Issue Active, ALU, FMA,
     FP16+Tensor, SFU pipes) and the instruction mix.
   - L1TEX highest → LSU/TEX-limited: check wavefronts and the L1 hit rate.
   - L2 highest → L2-bound: the working set spills out of L1; check L2 hit rate.
   - VRAM highest → bandwidth-bound: check the read/write split and sector hit rates.
3. If everything is low, you are latency-bound, launch-bound or idle: look at GR-engine activity,
   occupancy, and the range's duration rather than throughput.

Cross-check before believing a number: a 3 µs dispatch on a 1 TB/s GPU cannot have moved more
than ~3 MB, and a 1.3 ms scan that reads and writes 600 MB each has to be near the DRAM limit.

## SM occupancy and warp launch limits

- **SM Warp Occupancy** shows warp-slot residency over time, colored by shader type (VTG, PS,
  CS); it is the fastest way to see how much of the SM capacity a pass actually uses.
- **Per Shader Warp Occupancy** breaks the same data down per shader; the tooltip table lists
  Avg Warps and Occupancy % for the shaders in the selection. Use it to see which shader is
  keeping the SMs busy and which is starved.
- Low occupancy is only a problem if it comes with stalls. When a dispatch enqueues more work
  than fits on the SMs, the SMs report why additional warps **can't launch** (the
  [System Architecture Guide](https://docs.nvidia.com/nsight-graphics/UserGuide/gpu-trace-system-architecture.html)
  has the full deduction table):
  - compute warps: register allocation, warp allocation, CTA allocation, shared memory
    allocation;
  - 3D warps: register allocation, warp allocation, attribute allocation.
- The **Shader Pipelines** view has a `# Warp` column with the theoretical maximum warps per SM
  for that shader; hover it to see which resource limits it. Practical rules from the NVIDIA
  system-architecture guide: a CTA with 32 threads or fewer is limited to half occupancy
  (increase to 64); and shared memory per CTA times the number of CTAs competes with the per-SM
  budget (64 KiB compute-only mode, 32 KiB SCG mode).
- If the occupancy row shows warps but no visible launch limiter, either there is no launch
  stall (the warps are simply long-running) or the limiter is one of the allocation reasons the
  graph does not draw.

## Event List and Event Details

The Event List is a searchable/filterable subset of the API calls that matter for GPU
performance; selecting a row highlights it on the timeline, and Event Details shows its
parameters. Use it to confirm which submits are in the trace and to jump to one.
`--time-every-action` stops coalescing adjacent identical actions, at a small capture cost.

## Shader pipelines

The Shader Pipelines view is the primary shader-level table. Group rows by Shader Object,
Pipeline Object, or Shader Source. The columns that answer questions:

- **Type** — `Compute`, `Task`, `Mesh`, vertex/fragment/etc. The mesh example produces separate
  Task and Mesh rows.
- **Name / Hash / File Name** — the entry point, the bytecode hash, and the source file (with
  "Click to load" when debug info can be resolved).
- **`# Warp`** — theoretical warp occupancy; hover for the limiting resource.
- **`# Reg` / `Smem` / `CTA Dim`** — registers per thread, shared memory per thread group or per
  warp, and the thread-group size. These are the levers that decide occupancy.
- **Correlation** — whether debug/correlation info could be read.
- **Samples / Top Stall #1..#3** — with the Real-Time Shader Profiler, where the time went and
  why.
- **Instructions Executed / Thread Instructions Executed / Thread Instructions Executed Pred On /
  Active Threads Per Warp** — instruction counts and divergence. Active Threads Per Warp is
  pred-on threads over instructions times 32; a low value means branches are leaving lanes idle.
- Right-click actions: **Select on Timeline**, **Save as CSV…**, **Export bytecode…**.

"Unattributed" samples are correlation failures or internal GPU work; small counts are normal,
large counts are worth reporting to NVIDIA.

## Instruction mix and stall reasons

The **Instruction Mix** view breaks the selection down into instruction types (Self Instruction
Mix), the scoreboard producers it waits on (Input Dependencies), and the consumers of the
scoreboards it produces (Output Stall Locations). Memory loads, transcendentals and warp
primitives are producers; the stall appears on the consumer instruction, so a "long scoreboard"
on a line means the *load* feeding it is the culprit. NVIDIA's
[Shader Profiler reference](https://docs.nvidia.com/nsight-graphics/UserGuide/shader-profiler.html)
documents every column.

**Stall reasons** you will see in compute shaders, with the practical fix:

| Stall | Meaning | What to do |
|---|---|---|
| Long Scoreboard | Waiting on a local/global/texture load | Find the producer; improve locality/coalescing; if L1 hit rate is low or VRAM high, fix the access pattern |
| Short Scoreboard | Waiting on shared memory / MIO data | Check shared-memory bank conflicts and MIO pressure |
| Barrier | Waiting at a `GroupSync` | Reduce or narrow barriers; split large thread groups |
| Wait | Waiting on coupled math (FMA/ALU) | Normal dependency latency |
| Math Pipe Throttle | A math pipe FIFO is full | This is compute-bound; rebalance FP32/FP16 or move work |
| LG / MIO / TEX Throttle | An input FIFO is full | Too many memory instructions or local memory; vectorize accesses |
| Not Selected | Eligible, but another warp issued | Occupancy is sufficient; you have headroom for more work per thread |
| Drain | Exited warp waiting to drain memory writes | End-of-kernel effect |

The Shader Profiler is for **latency-bound** shaders: high SM activity/occupancy, low SM
throughput, low cache throughput. If SM throughput is high, you are compute-bound and should use
the throughput workflow instead.

## Shader source

The **Shader Source** tab shows high-level source (Slang), optionally side-by-side or interleaved
with lower-level code. Navigation tools: find, jump to highest/next-highest/lowest stall values,
a scrollbar heatmap, go-to-definition, and Top-Down/Bottom-Up call navigation. Hot Spots lists
the source lines with the most samples. All of it needs debug info in the SPIR-V
(`PICTOR_SHADER_DEBUG_INFO=1`) and, for sampled data, RTS.

## Reading GPUTRACE_REGIMES.xls

This export is the scriptable form of the GUI's metrics: one row per range, 123 columns with the
Async Compute Triage set. Column names have the shape
`<unit-group>.<metric-set>.<metric>.<aggregation>[.<modifier>]`, for example
`SM_A.TriageAC.l1tex__throughput.avg.pct_of_peak_sustained_elapsed`:

- the prefix (`SM_A`, `SM_B`, `SM_C`, `TPC`, `LTS`, `FBSP`, `PCI`, `FE_A`, `FE_B`, `GPC_A`,
  `GPC_B`, `ROP`, `HOST`) is the hardware block and collector instance that owns the counter;
- `TriageAC` is the Async Compute Triage metric set (screen-pipe metrics carry `TriageSCG`);
- `avg` is the average over the range, `sum` the total count, `pct` a ratio;
- `pct_of_peak_sustained_elapsed` compares to the block's peak sustained rate, averaged over the
  range's elapsed time; `per_cycle_elapsed` is the per-cycle rate; `peak_sustained` is the peak
  used for the percentage.

The row order matches `D3DPERF_EVENTS.xls` one-to-one, and nested ranges are flattened as
`parent/child` (`stream 0/scan32_compute`). The range duration can be recovered from
`gpc__cycles_elapsed.avg / (gpc__cycles_elapsed.avg.per_second * 1e9)` seconds
(`per_second` is in GHz in these exports).

The columns to look at first:

| Column (suffix) | Meaning |
|---|---|
| `TriageAC.sm__throughput.avg.pct_of_peak_sustained_elapsed` | SM issue throughput, % of peak |
| `SM_A.TriageAC.l1tex__throughput...` | L1TEX throughput, % of peak |
| `LTS.TriageAC.lts__throughput...` | L2 throughput, % of peak |
| `FBSP.TriageAC.dramc__throughput...` | DRAM throughput, % of peak (read + write) |
| `FE_B.TriageAC.gr__cycles_active...` | Fraction of the range the GR engine was active |
| `SM_B.TriageAC.tpc__ctas_active...` | Resident CTAs per TPC (and % of the 48-CTA peak this set uses) |
| `TPC.TriageAC.tpc__warps_active_shader_cs_realtime...` | Active compute warps, % of peak (`vtg` for mesh/geometry) |
| `TriageAC.tpc__warps_inactive_sm_active_realtime...` | Inactive warp slots while the SM has work (stalled) |
| `TriageAC.tpc__warps_inactive_sm_idle_realtime...` | Inactive warp slots while the SM is idle (no work) |
| `SM_A.TriageAC.sm__inst_executed_realtime.sum` | Instructions issued in the range |
| `SM_A...sm__inst_executed_pipe_alu_realtime.sum`, `SM_C...smsp__inst_executed_pipe_fma.sum`, `...pipe_fmaheavy`, `...pipe_xu` | Instruction mix by pipe (ALU, FMA/heavy, SFU) |
| `SM_A...l1tex__data_pipe_lsu_wavefronts...` | L1 LSU wavefront throughput |
| `SM_B.TriageAC.l1tex__t_sector_hit_rate.pct` | L1 hit rate |
| `LTS.TriageAC.lts__average_t_sector_hit_rate_realtime.pct` | L2 hit rate |
| `FBSP.TriageAC.dramc__read_throughput...` / `...write_throughput...` | DRAM read/write split |
| `PCI.TriageAC.pcie__read_bytes...` / `...write_bytes...` | PCIe traffic (uploads/downloads) |

Reasoning about compute- vs memory-bound:

- Compare the four throughputs (sm, l1tex, lts, dramc). The highest one is the candidate limiter;
  the P3 method is to fix the limiter, then re-measure.
- If SM is highest and near 100% → compute-bound; the instruction mix tells you which pipe.
- If L1TEX is highest → memory-instruction-bound at L1 (wavefronts, coalescing).
- If L2 is highest → the working set is spilling out of L1.
- If DRAM is highest → bandwidth-bound; read/write split and hit rates tell you why.
- If all four are low, read `gr__cycles_active` and the occupancy/warp counters: a range that is
  mostly idle (low GR activity) is launch-bound; a range with warps resident but not issuing
  (`warps_inactive_sm_active` high, `warps_active` moderate) is latency-bound on stalls.

Honest caveats, learned from the samples:

- **Tiny ranges are meaningless.** The repo binaries produce 3-13 µs ranges. Over such a range
  even a saturated unit averages to ~0% of the whole GPU's peak, so all percentages read as
  noise (our compute samples read 0.003-0.7%). The `sum` counters (instructions, bytes) and the
  durations are still readable, and so is structure (which dispatches exist, how they nest).
- **Sampled counters under-count.** In the `add_compute` sample the `dispatch_count` column reads
  0 for a range that visibly contains one dispatch. Cross-check anything suspicious against the
  event list.
- **Percentages are per-range averages, not peaks.** The GUI's metric graphs show the shape
  inside the range; the export only gives the average. A dispatch that saturates the SM for 10%
  of a range reads as 10%, not 100%.
- **Async Compute Triage is a triage set.** It answers "which unit is the candidate limiter",
  not "why" in detail; the GUI's Metrics tab lets you browse and filter the collected metrics per
  range, and Multi-Pass Metrics collects more (its Analysis view is Windows-only).
- **Clocks are locked to base** for reproducibility; absolute times differ from an unlocked run.

## A triage checklist

1. Trace Information: GPU, driver, metric set, clock lock; confirm the capture covers the pass
   you meant to trace.
2. Timeline: select the range for the pass (a user marker or a dispatch).
3. Summary tab: duration and the Unit Throughput table — the highest unit is the candidate
   limiter.
4. Metric graphs: is that unit's activity sustained or spiky? Was the GR engine active?
5. Occupancy: SM Warp Occupancy / Per Shader Warp Occupancy; if low, check the launch limits and
   the `# Warp` tooltip in Shader Pipelines.
6. With RTS (Ampere+): Shader Pipelines top stalls → Hot Spots → source. Long Scoreboard means
   memory latency; Math Pipe Throttle means compute-bound; Barrier means GroupSync.
7. If nothing is high anywhere, the workload is too small for the counters: make it bigger or
   trace a longer region before optimizing.
