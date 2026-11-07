# Troubleshooting

Known issues, driver/tool compatibility and GUI notes. For capture recipes see
[nsys.md](nsys.md) and [gputrace.md](gputrace.md).

## Tool and driver matrix

| Nsight Graphics | Required Linux driver | RTX 2070 SUPER (615.71.09) | RTX 4090 (595.71.05) |
|---|---|---|---|
| 2026.3 | 615 or newer | OK | **No** |
| 2026.2 | 580.126.18 or newer | OK | OK |
| 2026.1 | 580.126.18 or newer | OK | OK |
| 2025.5 | 580.105.08 or newer | OK | OK |

- GPU Trace metric sets (from `ngfx --help-all`): Ada has 0 - Async Compute Triage,
  1 - Throughput Metrics, 2 - Top-Level Triage; Turing has 0 - Throughput Metrics; Ampere GA10x
  has 0/1/2. The wrapper does not expose the selector; use `--metric-set-id`/`--architecture` on
  a manual `ngfx` run.
- Real-Time Shader Profiler: Ampere+ only. Turing gets the counter sets but not the sampling
  profiler.
- Nsight Systems: use 2026.3.1+ for Vulkan workload rows. Distribution packages are often far
  older (2024.x); use a current conda-forge install if the system one predates that.

## Counters permission

- Symptoms: "TARGET ERROR: GPU Performance Counters unavailable. Please enable access to GPU
  performance counters." (GPU Trace) or `ERR_NVGPUCTRPERM` (ncu).
- On R610+ the capability nodes (`profiler-device`, `profiler-context`, `trace-device`) are the
  gate; grant them with the script in [gputrace.md](gputrace.md#counter-permissions-linux) or run
  the tool as root. Pre-R610 drivers are admin-only (`RmProfilingAdminOnly: 1`), so counter traces
  run under `sudo` there.
- Without the grant the wrapper still produces a valid event/label trace (`--no-metrics` forces
  that path).

## Nsight Systems issues

| Symptom | Cause / fix |
|---|---|
| No labels on the timeline | Captured with `--trace=vulkan`; use `vulkan-annotations` |
| `VULKAN_WORKLOAD` empty / no GPU rows | nsys older than 2026.3.1 (2026.2 silently records none); or `--vulkan-gpu-workload=none` |
| `vulkan_gpu_marker_sum` empty | Captured with `--vulkan-gpu-workload=batch` (loses labels); use `individual` |
| Cannot attach to a running app | Not supported on Linux; launch the target under `nsys` |
| Missing `vkCmdDispatch` etc. | nsys records only a subset of `vkCmd*` calls; use debug ranges for dispatch structure |
| Barrier counts look low | `vkCmdPipelineBarrier2` is recorded only after another barrier; treat counts as a lower bound |
| CPU sampling fails | Restricted `perf_event_open`; add `--sample=none` |
| Last submit has no workload row | Short-lived app exiting right after the last submit; keep it alive, or use `VULKAN_DEBUG_API` |
| Slower than normal / noisy numbers | Validation layers enabled; run performance captures without them |

## GPU Trace issues

| Symptom | Cause / fix |
|---|---|
| `ngfx --exe` crashes the target (SIGSEGV in `libGLX_nvidia.so` during ICD negotiation) | Known launcher issue on some drivers; use `scripts/gputrace.sh` (attach mode) |
| "Process activity doesn't match" on attach | The target is missing the `NV_TPS_LAUNCH_TOKEN` / `GPUTRACE_*` block; the wrapper sets it |
| Counters unavailable | Permission grant missing; run under `sudo` or apply the capability script |
| App appears to hang before the first submit | Expected with counters enabled: the layer holds the target until the host session is up. Attach the host, or run with `--no-metrics` |
| SDK-stopped trace aborts after writing the report (driver 595, counters on) | Known layer teardown issue; the report is valid. Use submit scoping for counters, or SDK without `--metrics` |
| `--rts` seems ignored | With the default Async Compute Triage set on Ada, `REPRO_INFO.xls` still reports "Real-Time Shader Profiler: Disabled" even though the host log shows warp-state sampling configured; all metric sets tried (0/1/2) behaved the same in attach mode. Verify in the GUI's Trace Information / Shader Profiler tabs, and try `--metric-set-id` explicitly. Treat the export's RTS field as unreliable. |
| No metrics in the export | Captured with `GPUTRACE_DISABLE_METRICS=1` (wrapper degradation) or counters unavailable; `GPUTRACE_REGIMES.xls` then has only the `flattened_event_name` column |
| Trace window misses kernels or is empty | `--start-after-submits N` skips the first N submits and `--limit-to-submits M` stops at submit M; the window is (N, M]. Set M to the index of the last submit you want (over-capturing is harmless) |
| Shader rows empty | Captured with `--disable-collect-shader-pipelines`; re-enable (default on) |
| `Missing session` / `Disconnecting... Failed` after the report is saved | The target exited before the host disconnected; harmless — the report and exports are complete |
| Numbers vary between runs | Clocks are locked to base for comparability; other GPU activity still pollutes (GPU Trace collects all GPU activity). Run on an idle GPU. |

## Source view and debug info

- The source view needs debug info in the SPIR-V: build with `PICTOR_SHADER_DEBUG_INFO=1` (or set
  the env override). Without it, the Shader Pipelines/Flame Graph/Top-Down views have no
  function data.
- Paths are virtual by default (`/pictor/jit.slang`, `/sp0/add.slang`); the full source text is
  embedded, so the tools can display it, but "Click to load"/external navigation needs real
  files. Set `PICTOR_SOURCE_ROOT=<dir>` to write the closure to disk and embed real paths, or
  add search paths under Tools → Options → Search Paths in the GUI.
- NVIDIA notes that Slang's NonSemantic output "generally provides reliable function and
  call-site information"; aggressive inlining can still blur call-site attribution.

## GUI notes

- The Nsight Graphics GUI and `nsys-ui` need a display; the captures themselves are headless and
  the reports are portable (open them on any machine with the tool installed).
- The GPU Trace **Analysis view** (multi-pass metrics with automatic recommendations) is
  Windows-only. On Linux, use the Metrics tab and the tables.
- "Save as CSV…" on the Shader Pipelines/Event List tables is the supported way to get data out
  of the GUI; the `BASE/*.xls` exports cover scripted checks.
- For multi-frame reports, the Aggregate Frames option averages metrics across frames and greys
  out values with high variance; frameless traces have a single frame.
- Trace Compare (two reports side by side) is a quick way to see a change; align by marker.

## NVIDIA documentation

- Nsight Systems User Guide: https://docs.nvidia.com/nsight-systems/UserGuide/index.html
  (Vulkan API trace, CLI flags, capture ranges)
- Nsight Systems Analysis Guide: https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html
  (Vulkan reports, SQLite schema)
- GPU Trace overview: https://docs.nvidia.com/nsight-graphics/UserGuide/gpu-trace-overview.html
  (frameless workflow, CLI options, memory settings)
- GPU Trace UI reference: https://docs.nvidia.com/nsight-graphics/UserGuide/gpu-trace-ui.html
  (timeline rows, Event List, metrics tab, occupancy rows, exports)
- Shader Profiler: https://docs.nvidia.com/nsight-graphics/UserGuide/shader-profiler.html
  (Shader Pipelines, stall reasons, instruction mix, source view)
- System Architecture Guide: https://docs.nvidia.com/nsight-graphics/UserGuide/gpu-trace-system-architecture.html
  (unit throughputs, warp can't launch reasons, SM/L1/VRAM reading)
- Configure application (markers, Slang debug info): https://docs.nvidia.com/nsight-graphics/UserGuide/configure-application.html
- Nsight Graphics SDK: https://docs.nvidia.com/nsight-graphics/UserGuide/sdk.html
- Nsight Graphics Release Notes (driver matrix, Turing deprecation): https://docs.nvidia.com/nsight-graphics/ReleaseNotes/index.html
- Counter permissions (`ERR_NVGPUCTRPERM`): https://developer.nvidia.com/ERR_NVGPUCTRPERM
- Peak-Performance-Percentage analysis method: https://developer.nvidia.com/blog/the-peak-performance-analysis-method-for-optimizing-any-gpu-workload/
