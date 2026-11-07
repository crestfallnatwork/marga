# GPU profiling with NVIDIA tools

pictor_core is built for performance work: the runtime emits `VK_EXT_debug_utils` names and ranges
and can embed Slang debug info, so NVIDIA's tools see named kernels, queues and objects instead of
anonymous Vulkan calls. Two tools cover the ground:

- **Nsight Systems (`nsys`)** — the timeline: CPU/GPU scheduling, the Vulkan API trace, debug
  ranges, sync analysis.
- **Nsight Graphics GPU Trace** — hardware counters and shader analysis: unit throughput (SM,
  L1TEX, L2, VRAM), occupancy, warp-launch limits, stall reasons, instruction mix, shader source.

Nsight Compute (`ncu`) is CUDA-only and cannot profile Vulkan kernels; it is not part of this
workflow.

Everything in these guides was verified against captures on RTX 2070 SUPER (driver 615) and
RTX 4090 (driver 595) systems. Where behavior depends on the GPU, the driver or counter
permissions, it is called out.

## Getting started: trace a three-kernel program

This is the shape most pictor_core workloads have: two compute kernels and one mesh kernel,
executed in sequence, then exit. The runtime emits everything the tools need — no source changes.

**The program** (abridged; `examples/` has runnable versions):

```cpp
add->record(kAddCount, 1, 1, add_params);     // -> labeled "add_compute"
ctx->fsync();                                 // one submit
add->record(kAddCount, 1, 1, add_params);     // second compute kernel
ctx->fsync();
tile->record(kTileCount, 1, 1, tile_params);  // -> task + mesh draw, "tile_mesh"
ctx->fsync();
```

Each `record()` + `fsync()` is one submit. The runtime wraps every dispatch/draw in a debug-utils
range named after its shader entry point, labels each submit (`stream 0`), and names pipelines, so
the tools show named work everywhere.

**Step 1 — the timeline (nsys).** nsys launches the app itself and records every submit and label:

```bash
nsys profile --trace=vulkan-annotations --vulkan-gpu-workload=individual \
  --stats=true --force-overwrite=true -o trace ./build/your_app
```

`--stats=true` prints a GPU marker summary with one row per label; open `trace.nsys-rep` in the
Nsight Systems GUI for the full timeline. No elevated permissions are needed. Reference:
[profiling/nsys.md](profiling/nsys.md).

**Step 2 — the counters (GPU Trace).** GPU Trace attaches to the running app and collects hardware
counters for a window of submits:

```bash
sudo scripts/gputrace.sh ./build/your_app \
  --start-after-submits 0 --limit-to-submits 8 \
  --metrics --out gputrace/three-kernels
```

The window is expressed in submits: `--start-after-submits N` skips the first N submits and
`--limit-to-submits M` stops at submit M, so the trace covers submits N+1 through M. Over-capturing
is harmless; set M to the index of the last submit you want. With counters enabled the GPU Trace
layer holds the app at its first submit until the host session is up, so short-lived programs trace
reliably. Counters need elevated permissions on most drivers — see
[profiling/gputrace.md](profiling/gputrace.md) for the grant and the no-permission fallback.

The export lists one named range per kernel:

```text
event_text      time_ms
stream 0        0.011392
        add_compute     0.011392
stream 0        0.011488
        add_compute     0.011488
stream 0        0.361152
        tile_mesh       0.361152
```

Open the `.ngfx-gputrace` report in the Nsight Graphics GUI to see the timeline, per-range metrics
and the shader pipelines (Compute, Task and Mesh stages). The full walkthrough, including the
submit-window experiments behind that output, is in [profiling/tutorials.md](profiling/tutorials.md).

## Runtime instrumentation

Three CMake options control what the runtime emits; all can be overridden per run:

| CMake option | Default | Runtime override | Effect |
|---|---|---|---|
| `PICTOR_ENABLE_DEBUG_UTILS` | `ON` | `PICTOR_DEBUG_UTILS=0/1` | Enables `VK_EXT_debug_utils`; names objects and records ranges |
| `PICTOR_ENABLE_SHADER_DEBUG_INFO` | `OFF` | `PICTOR_SHADER_DEBUG_INFO=0/1` | Compiles Slang shaders with NonSemantic debug info (`-g2`) |
| `PICTOR_ENABLE_NGFX_SDK` | `OFF` | `PICTOR_NGFX_SDK=1` | Compiles in the Nsight Graphics SDK one-shot trace hooks |

Names and ranges are emitted automatically for:

- the queue around each submit (`stream N`) and each command buffer (`stream N cmd M`),
- every executor dispatch/draw — the range carries the shader entry name (`scan64_compute`,
  `add_compute`, ...),
- pipelines — named after the entry point.

Application code can add its own structure:

```cpp
stream->begin_label("build pass");
// ... executors record work ...
stream->end_label();
ctx->set_name(some_buffer, "keys");
```

Turning everything off: `PICTOR_DEBUG_UTILS=0` — the extension is not enabled, no ranges are
recorded, no overhead.

Two more environment variables matter for the tools:

- `PICTOR_SHADER_DEBUG_INFO=1` embeds the full shader source and NonSemantic debug info in the
  SPIR-V, which is what the GPU Trace Shader Profiler's source view consumes. The env override
  works even in a build configured with the option off.
- `PICTOR_SOURCE_ROOT=<dir>` writes the shader closure to `<dir>` (for example
  `<dir>/sp0/add.slang`, `<dir>/pictor/jit.slang`) and embeds those real paths, so the tools can
  resolve files on disk instead of the virtual paths.

For performance numbers, run without the validation layers
(`VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation VK_LOADER_LAYERS_ENABLE='*validation'` is the
validation-clean setup, but it adds overhead and API noise).

## Where to go next

| Document | Contents |
|---|---|
| [timing.md](timing.md) | In-app GPU phase timing with `GpuTimer`: laps, nested phases, readback and contracts |
| [profiling/nsys.md](profiling/nsys.md) | Nsight Systems: install, capture recipes, reports, SQLite queries, limitations |
| [profiling/gputrace.md](profiling/gputrace.md) | GPU Trace: requirements, counter permissions, capture windows, the SDK path, exports |
| [profiling/reading-gputrace.md](profiling/reading-gputrace.md) | Reading a GPU Trace report: timeline, unit throughput, occupancy, stalls, shader source |
| [profiling/tutorials.md](profiling/tutorials.md) | Worked examples: the three-kernel program, scan, mesh, a calibrated memory-bound sample |
| [profiling/troubleshooting.md](profiling/troubleshooting.md) | Tool/driver matrix, known issues, GUI notes, NVIDIA documentation links |
