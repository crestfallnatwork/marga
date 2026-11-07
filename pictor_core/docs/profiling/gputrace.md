# Nsight Graphics GPU Trace

GPU Trace is the counter profiler: unit throughput (SM, L1TEX, L2, VRAM), warp occupancy,
warp-launch limits, stall reasons, instruction mix, and shader source. Headless compute workloads
work fine — `scripts/gputrace.sh` starts the app with the GPU Trace layer and attaches the host,
scoping the trace by submits.

Start with [the getting-started guide](../profiling.md); to interpret a report, see
[reading-gputrace.md](reading-gputrace.md).

## Requirements

- Nsight Graphics 2026.2 or newer. Linux driver: 2026.2 needs 580.126.18+, 2026.3 needs 615+.
- Hardware counters need elevated permissions on most drivers (below); without them the wrapper
  produces an event/label-only trace.
- The Real-Time Shader Profiler needs Ampere or newer.
- Turing support is deprecated and will be removed in a future release.

## Counter permissions (Linux)

Symptoms: "TARGET ERROR: GPU Performance Counters unavailable. Please enable access to GPU
performance counters." (GPU Trace) or `ERR_NVGPUCTRPERM` (ncu).

On R610+ drivers the capability nodes (`profiler-device`, `profiler-context`, `trace-device`) are
the gate. As root, grant them to all users (use `setfacl` for a single user):

```bash
for cap in profiler-device profiler-context trace-device; do
    minor=$(awk -v c="$cap" '$1==c{print $2}' /proc/driver/nvidia-caps/sys-minors)
    nvidia-modprobe -f /proc/driver/nvidia/capabilities/$cap
    chmod a+r /dev/nvidia-caps/nvidia-cap$minor
    echo "DeviceFileModify: 0" > /proc/driver/nvidia/capabilities/"$cap"
done
```

Alternatives: run under `sudo`, grant the user `CAP_SYS_ADMIN` or `CAP_PERFMON` (driver R565+), or
the legacy module option `NVreg_RestrictProfilingToAdminUsers=0` + reboot (slated for removal).
Pre-R610 drivers predate the capability nodes and are admin-only, so counter traces run under
`sudo` there.

The wrapper probes the capability nodes and degrades gracefully: without permission it prints a
notice and produces an event/label-only trace (`--no-metrics` forces that path).

## Capture with the wrapper

```bash
sudo scripts/gputrace.sh <app> [app args...] \
  [--start-after-submits N] [--limit-to-submits M] [--max-duration-ms D] \
  [--out DIR] [--metrics] [--no-metrics] [--rts] [--sdk]
```

The wrapper launches the app with the GPU Trace layer enabled (it does not use `ngfx --exe`, whose
launcher can crash the target during Vulkan ICD negotiation on some drivers), attaches `ngfx` to
the running process, and exports into the output directory. Defaults: start after 0 submits, limit
8 submits, 2000 ms max duration.

Examples:

```bash
# Event/label-only trace (no counters needed)
scripts/gputrace.sh ./build/algorithms/pictor_scan_test --limit-to-submits 8

# Counter trace (root or the capability grant)
sudo scripts/gputrace.sh ./build/algorithms/pictor_scan_test \
  --limit-to-submits 8 --metrics --out gputrace/scan
sudo chown -R $USER gputrace/scan     # report files are root-owned

# Real-Time Shader Profiler (Ampere+)
sudo scripts/gputrace.sh ./build/examples/mesh/example_mesh --limit-to-submits 2 --metrics --rts
```

### The submit window

`--start-after-submits N` skips the first N submits and `--limit-to-submits M` stops at submit M,
so the trace covers **submits N+1 through M** (1-indexed; all submits count, including
transfer-only ones). Verified on a 4-submit app (uploads + two compute kernels + one mesh kernel):

| `--start-after-submits` / `--limit-to-submits` | Captured |
|---|---|
| `0` / `4` (or any M >= 4) | all three kernels |
| `1` / `3` | submits 2-3 |
| `1` / `2` | submit 2 only |

Set M to the index of the last submit you want; over-capturing is harmless (the trace is bounded
by `--max-duration-ms`). To find the indices, run the app once under nsys (the marker summary
lists every labeled dispatch, see [nsys.md](nsys.md)), or capture with a generous limit and read
`D3DPERF_EVENTS.xls`.

### Short-lived programs

With counters enabled the GPU Trace layer synchronizes the target with the host: the app **blocks
at its first submit** until the host session is established (verified: a 4-submit app hangs at
submit 1 when no host is attached, and proceeds as soon as `ngfx` attaches). A program that runs a
few kernels and exits is therefore traced reliably — the attach delay is not a race.

Metrics-disabled (event-only) mode does not hold the target, so a program that finishes before
the host attaches may not be traced. For short programs without counters, either keep the process
alive (loop the pass) or use the SDK path below, which waits for the profiler.

### Expected output

```text
Preparing to collect a GPU Trace report...
Generating a trace after 0 submits with a max duration of 2000ms, limited to 8 submits...
GPU Trace report saved to gputrace/three-kernels/three_kernels_2026_10_13_16_40_08.ngfx-gputrace
Opening generated GPU Trace report...
Succeeded to export data:
    gputrace/three-kernels/BASE/REPRO_INFO.xls
    gputrace/three-kernels/BASE/D3DPERF_EVENTS.xls
    gputrace/three-kernels/BASE/FRAME.xls
    gputrace/three-kernels/BASE/GPUTRACE_FRAME.xls
    gputrace/three-kernels/BASE/GPUTRACE_REGIMES.xls
Disconnecting...
gputrace: report  -> gputrace/three-kernels/three_kernels_2026_10_13_16_40_08.ngfx-gputrace
gputrace: exports -> gputrace/three-kernels/BASE*/
```

The report files are root-owned when the wrapper runs under `sudo`; `chown` them before opening
them in the GUI as a user.

## App-driven one-shot capture (SDK)

Submit scoping cannot express "trace exactly this pass" (or "trace every iteration of a loop").
For that, compile with `-DPICTOR_ENABLE_NGFX_SDK=ON` (override `PICTOR_NGFX_SDK_INCLUDE_DIR` if
the SDK is not under `/opt/nsight-graphics/latest`) and bracket the pass:

```cpp
{
    pictor::gputrace::ScopedRange range;
    exec->record(...);
    ctx->fsync();
}
```

`Context::create()` initializes the SDK automatically when the app runs with `PICTOR_NGFX_SDK=1`
and a GPU Trace session is attached; otherwise `ScopedRange` is a no-op. When opted in, the first
`ScopedRange` waits up to ~5 s for the profiler to attach before starting the trace. Run the
wrapper with `--sdk`:

```bash
sudo scripts/gputrace.sh ./build/examples/compute/example_compute --sdk --no-metrics
```

The wrapper sets the target's start/stop conditions to the SDK events and the host to
`--start-with-ngfx-sdk`/`--stop-with-ngfx-sdk`; the trace covers exactly what the scoped block
submits. `examples/compute/add.cpp` shows the pattern. The export shows the queue and dispatch
labels:

```text
event_text	time_ms
stream 0	0.002496
        add_compute	0.002496
```

The SDK is beta (0.9.0, header-only), and only one of GPU Trace / Graphics Capture can be
injected into a process at a time.

### Why the runtime loads the injection library itself

The SDK ships as **headers only** (no linkable library): its functions dispatch through a table
that the tool's injection library (`libWarpViz.Injection.so`) populates at runtime. The supported
way to get that library into the process is launching through `ngfx` — but the launch path loads
the injection library before Vulkan initialization, which crashes the target during ICD
negotiation on some drivers (SIGSEGV in `libGLX_nvidia.so`). On Linux there is no self-injection
API (`NGFX_GPUTrace_Inject_Vulkan` is Windows-only), so `pictor::gputrace::initialize()` loads
the library itself — after Vulkan initialization, and reusing an already-loaded copy when the
tool injected it. That is the attach-mode path the wrapper drives.

Known issue: on driver 595 with counters enabled, an SDK-stopped trace has been observed to abort
the process after the report is written (layer teardown, `double free`); the report is still valid.
Use submit scoping for counter traces there, or run the SDK path without `--metrics`.

## Without the wrapper

The wrapper is the supported path, but the underlying attach command is worth knowing. The layer
is enabled through environment variables, the app is launched by hand, and the host attaches to
its PID:

```bash
INST=/opt/nsight-graphics/latest
LAYER_DIR=$INST/target/linux-desktop-nomad-x64/GPUTraceVulkanLayer

env \
  ENABLE_VK_LAYER_NV_GPU_Trace_release_public_2026_2_0=1 \
  XDG_DATA_DIRS="/usr/local/share:/usr/share:$LAYER_DIR" \
  NV_TPS_LAUNCH_TOKEN="GPU Trace Profiler" \
  GPUTRACE_START_EVENT="Submit Count" GPUTRACE_START_SUBMIT_COUNT=0 \
  GPUTRACE_STOP_EVENT="Max Submits" GPUTRACE_STOP_SUBMIT_COUNT=2 \
  GPUTRACE_STOP_DURATION_IN_MS=2000 \
  ./build/examples/mesh/example_mesh &
APP_PID=$!
sleep 1
$INST/host/linux-desktop-nomad-x64/ngfx --activity="GPU Trace Profiler" \
  --platform="Linux (x86_64)" --attach-pid=$APP_PID \
  --start-after-submits=0 --limit-to-submits=2 --max-duration-ms=2000 \
  --auto-export --output-dir="$PWD/gputrace-mesh"
wait $APP_PID
```

Notes:

- The enable variable name is version-specific (it comes from the layer's JSON file); the wrapper
  discovers it automatically.
- The full `GPUTRACE_*` block matters: with only the layer enabled, attach fails with
  "Process activity doesn't match".
- The host CLI start/limit conditions override the target env in attach mode.
- `--metric-set-id` / `--architecture` select the metric set; `--real-time-shader-profiler`
  requests RTS (with a caveat, see [troubleshooting.md](troubleshooting.md)).

## Exports

With `--auto-export`, a successful trace produces:

```text
<out>/<app>_<date>_<time>.ngfx-gputrace     # proprietary binary report ("WRPV" magic)
<out>/ReportGeneratorTags.txt               # e.g. "GPU Trace Data | BASE"
<out>/BASE/REPRO_INFO.xls                   # machine/GPU/driver/process info + trace settings
<out>/BASE/D3DPERF_EVENTS.xls               # event_text, time_ms (tab-separated)
<out>/BASE/FRAME.xls                        # GPU frame time (empty for frameless traces)
<out>/BASE/GPUTRACE_FRAME.xls               # metric rows in table form
<out>/BASE/GPUTRACE_REGIMES.xls             # per-range metrics (the interpretation workhorse)
```

The `.xls` files are **plain tab-separated text**. Nested events in `D3DPERF_EVENTS.xls` are
indented children of the row above (eight spaces); `GPUTRACE_REGIMES.xls` flattens the same
hierarchy with `/` (`stream 0/init_compute`) and its rows are in the same order as the event
list, one row per range. The GUI additionally offers "Save as CSV…" on the GPU Trace and Shader
Profiler tables.

The report is portable: open it offline on any machine with Nsight Graphics installed.

## Source correlation

Compile with `PICTOR_SHADER_DEBUG_INFO=1` (or the CMake option) so the SPIR-V carries NonSemantic
debug info, and set `PICTOR_SOURCE_ROOT=<dir>` to write the shader closure to disk and embed real
paths — the Shader Profiler's source view then resolves files. The full source text is embedded in
the SPIR-V either way; without a source root the paths are virtual (`/pictor/jit.slang`,
`/sp0/add.slang`) and the tools show the embedded text but cannot open the files. NVIDIA's docs
state that Slang's NonSemantic output "generally provides reliable function and call-site
information"; aggressive inlining can still blur call-site attribution.

To pass the variables through the wrapper (which runs the app itself):

```bash
sudo env PICTOR_SOURCE_ROOT=$PWD PICTOR_SHADER_DEBUG_INFO=1 \
  scripts/gputrace.sh ./build/your_app --limit-to-submits 8 --metrics
```
