# Nsight Systems (`nsys`)

Nsight Systems is the timeline tool: CPU/GPU scheduling, the Vulkan API trace, debug-utils ranges
and sync analysis. It launches the target itself (no attach on Linux) and needs no elevated
permissions. It answers "what ran when, on which queue, and how long" — for hardware counters and
shader analysis, use [GPU Trace](gputrace.md).

Start with [the getting-started guide](../profiling.md) if you have not traced anything yet.

## Install

```bash
# conda-forge (no root; includes the CLI and the nsys-ui GUI).
# The libmamba solver currently fails to resolve nsight-systems; add --solver classic.
conda install -c conda-forge --solver classic nsight-systems nsight-systems-cli

# Ubuntu
sudo apt install nsight-systems

# Arch
sudo pacman -S nsight-systems
```

Use **nsys 2026.3.1 or newer** for Vulkan GPU workload rows. nsys 2026.2 silently records none,
and distribution packages are often far older than that.

## Capture

```bash
nsys profile \
  --trace=vulkan-annotations \
  --vulkan-gpu-workload=individual \
  --stats=true --force-overwrite=true \
  -o scan \
  ./build/algorithms/pictor_scan_test
```

- `vulkan-annotations` traces the Vulkan API **and** collects debug-utils labels; plain `vulkan`
  collects no labels.
- `--vulkan-gpu-workload=individual` (the default) records GPU workload rows: one per submit plus
  one per debug-utils range, the latter carrying the label text (`init_compute`, ...). `batch`
  collapses everything into one coarse row without labels — keep `individual`.
- `--stats=true` runs the default summary reports (the two marker reports) after collection.
- `--force-overwrite=true` allows re-running into the same output name.
- On Linux the target must be launched by `nsys`; attaching to an already-running Vulkan app is
  not supported.
- If `perf_event_open` is restricted, add `--sample=none`; CPU IP sampling is not needed for the
  Vulkan reports.

Expected tail of the run (RTX 2070 SUPER, driver 615, nsys 2026.3.2):

```text
 ** Vulkan Range Summary (vulkan_marker_sum):

 Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)      Range
 --------  ---------------  ---------  --------  --------  --------  --------  -----------  --------------
     78.7          620,510         48  12,927.3   3,141.0     2,204   315,842     45,061.4  stream 0
     15.1          118,926         48   2,477.6     907.0       672    11,392      2,945.2  init_compute
      3.2           25,531         24   1,063.8     902.0       661     2,384        416.3  scan32_compute
      3.0           23,595         24     983.1     851.5       681     2,124        360.7  scan64_compute

 ** Vulkan GPU Range Summary (vulkan_gpu_marker_sum):

 Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)      Range
 --------  ---------------  ---------  --------  --------  --------  --------  -----------  --------------
     58.1        1,156,733         24  48,197.2  12,209.0    11,873   382,179    103,592.7  scan64_compute
     34.2          680,091         24  28,337.1  12,593.0    10,818   177,953     46,058.4  scan32_compute
       7.7         152,720         48   3,181.7   3,393.0     1,473     4,545        728.2  init_compute
```

The two reports answer different questions. `vulkan_marker_sum` measures the **CPU** ranges: how
long the record/submit path took on the host. `vulkan_gpu_marker_sum` measures the **GPU** ranges:
how long each labeled dispatch actually ran on the device. For the scan test the CPU view says
`init_compute` is 15% of label time (it runs 48 times), while the GPU view says `scan64_compute`
dominates at 58% (it is the big one). Use the GPU report to find expensive dispatches.

## Reports

```bash
nsys stats --report vulkan_marker_sum --report vulkan_gpu_marker_sum scan.nsys-rep
nsys stats --report vulkan_api_sum scan.nsys-rep
nsys export --type sqlite --force-overwrite=true -o scan.sqlite scan.nsys-rep
```

The shipped Vulkan reports are `vulkan_api_sum`, `vulkan_api_trace`, `vulkan_gpu_marker_sum` and
`vulkan_marker_sum`; only `vulkan_gpu_marker_sum` reads the raw workload rows (and only those with
label text), so raw per-submit rows are visible only in the GUI timeline or in SQLite.

`vulkan_api_sum` (trimmed) for the scan test:

```text
 Time (%)  Total Time (ns)  Num Calls    Avg (ns)   ...              Name
 --------  ---------------  ---------  ------------  ...  ----------------------------
     39.0       64,876,062          1  64,876,062.0  ...  vkCreateDevice
     26.7       44,364,894         97     457,370.0  ...  vkWaitForFences
     12.2       20,297,438          1  20,297,438.0  ...  vkDestroyDevice
     11.2       18,708,357         13   1,439,104.4  ...  vkAllocateMemory
      8.1       13,559,810         13   1,043,062.3  ...  vkFreeMemory
      1.2        1,935,949         48      40,332.3  ...  vkBeginCommandBuffer
      0.7        1,227,912          3     409,304.0  ...  vkCreateComputePipelines
      0.3          500,551         48      10,428.1  ...  vkQueueSubmit
      0.0           27,292         96         284.3  ...  vkCmdBindPipeline
      0.0           13,626         48         283.9  ...  vkCmdPipelineBarrier2
```

## SQLite queries

The SQLite export is where the Vulkan-specific detail lives. Useful tables: `VULKAN_WORKLOAD`
(GPU workload rows; the label text is `textId`), `VULKAN_DEBUG_API` (our labels),
`VULKAN_API` (the recorded call subset), `VULKAN_MEMORY_OPERATION` / `VULKAN_MEMORY_TYPES`,
`VULKAN_PIPELINE_CREATION_EVENTS` and `VULKAN_PIPELINE_STAGE_EVENTS` (pipeline feedback).

**Workload rows** — one per submit (`vkQueueSubmit`) and one per debug-utils label, correlated to
the label-begin API call:

```bash
sqlite3 scan.sqlite "SELECT (SELECT value FROM StringIds WHERE id=nameId) AS name,
  (SELECT value FROM StringIds WHERE id=textId) AS label, COUNT(*)
  FROM VULKAN_WORKLOAD GROUP BY nameId, textId;"
```

```text
vkQueueSubmit                                 NULL             48
vkCmdBeginDebugUtilsLabelEXT                  init_compute     48
vkCmdBeginDebugUtilsLabelEXT                  scan32_compute   24
vkCmdBeginDebugUtilsLabelEXT                  scan64_compute   24
```

GPU time per label (the raw form of `vulkan_gpu_marker_sum`):

```bash
sqlite3 scan.sqlite "SELECT (SELECT value FROM StringIds WHERE id=textId) AS label,
  COUNT(*) AS n, ROUND(AVG(end - start)/1000.0, 1) AS avg_us,
  ROUND(SUM(end - start)/1000.0, 1) AS total_us
  FROM VULKAN_WORKLOAD WHERE textId IS NOT NULL GROUP BY textId ORDER BY total_us DESC;"
```

```text
scan64_compute  24    48.2    1156.7
scan32_compute  24    28.3     680.1
init_compute    48     3.2     152.7
```

**Debug API** — the label begin/end calls and queue labels:

```bash
sqlite3 scan.sqlite "SELECT (SELECT value FROM StringIds WHERE id=nameId) AS call,
  (SELECT value FROM StringIds WHERE id=textId) AS label, COUNT(*)
  FROM VULKAN_DEBUG_API GROUP BY nameId, textId;"
```

```text
vkCmdBeginDebugUtilsLabelEXT    init_compute     48
vkCmdBeginDebugUtilsLabelEXT    scan32_compute   24
vkCmdBeginDebugUtilsLabelEXT    scan64_compute   24
vkCmdEndDebugUtilsLabelEXT                      96
vkQueueBeginDebugUtilsLabelEXT  stream 0         48
vkQueueEndDebugUtilsLabelEXT                    48
```

**Memory operations** — host-blocking writes/reads and bind/map/unmap ranges:

```bash
sqlite3 scan.sqlite "SELECT (SELECT value FROM StringIds WHERE id=v.nameId) AS call,
  COUNT(*) AS n, SUM(COALESCE(m.memorySize,0)) AS total_bytes
  FROM VULKAN_MEMORY_OPERATION m JOIN VULKAN_API v ON v.id = m.traceEventId
  GROUP BY v.nameId ORDER BY n DESC;"
```

```text
vkBindBufferMemory  240    874707648
vkFreeMemory         13            0
vkAllocateMemory     13            0
vkMapMemory           9            0
vkUnmapMemory         4            0
```

`VULKAN_MEMORY_TYPES` maps the allocations to heaps (`VulkanGpuDefaultHeap`,
`VulkanCpuSharedHeap`); this is the data behind the GUI's "Vulkan Memory Operations" row.

**Pipeline creation and stage events** — driver pipeline-creation feedback (driver 435+):

```bash
sqlite3 scan.sqlite "SELECT (SELECT value FROM StringIds WHERE id=v.nameId) AS call,
  c.duration AS duration_ns, c.flags
  FROM VULKAN_PIPELINE_CREATION_EVENTS c JOIN VULKAN_API v ON v.id = c.traceEventId;"
sqlite3 scan.sqlite "SELECT duration, flags FROM VULKAN_PIPELINE_STAGE_EVENTS;"
```

```text
vkCreateComputePipelines      1012611      1
vkCreateComputePipelines       166588      1
vkCreateComputePipelines        42340      1

822759      1
151159      1
 30468      1
```

**Correlating a GPU range to the CPU call** — `VULKAN_WORKLOAD.correlationId` joins
`VULKAN_API.correlationId`:

```bash
sqlite3 ex.sqlite "SELECT w.correlationId,
  (SELECT value FROM StringIds WHERE id=w.textId) AS label,
  (SELECT value FROM StringIds WHERE id=v.nameId) AS api_call, v.start, v.end
  FROM VULKAN_WORKLOAD w LEFT JOIN VULKAN_API v ON v.correlationId = w.correlationId;"
```

```text
4               vkQueueSubmit                 270889365  271297042
2  add_compute  vkCmdBeginDebugUtilsLabelEXT  270868224  270868896
2  add_compute  vkCmdEndDebugUtilsLabelEXT    270880047  270880207
```

The CPU and GPU timestamps are in different clock domains — compare durations, not absolute times.

## Reading the timeline

- Each queue gets a row with an **API sub-row** (where `vkQueueSubmit` was called) and a **GPU
  Workload sub-row** (GPU execution). Clicking one highlights the other.
- Debug-utils labels appear on both the CPU and the GPU side (with the annotations collector);
  our `stream N` queue labels and per-dispatch labels are what make the timeline readable.
- The **Vulkan Memory Operations** row shows host-blocking writes/reads and map/unmap ranges.
- Pipeline creation feedback shows per-stage durations, which is how you catch a shader that is
  expensive to compile.
- The CPU side is dominated by `vkWaitForFences` whenever the app blocks after each case (as the
  scan test does). Do not read those gaps as GPU problems; read the workload rows.

## Limitations

- nsys traces only a selected subset of Vulkan API calls. Recorded: `vkCreate*`/`vkDestroy*`,
  `vkBeginCommandBuffer`/`vkEndCommandBuffer`, `vkQueueSubmit`, `vkCmdBindPipeline`,
  `vkCmdPipelineBarrier2` (partial, see below), the debug-utils label calls, and the memory
  calls. **Not recorded** although the app calls them: `vkCmdDispatch`, `vkCmdPushConstants`,
  `vkCmdPushDescriptorSetKHR`, `vkCmdCopyBuffer`, `vkCmdFillBuffer`, `vkCmdUpdateBuffer`,
  `vkCmdDrawMeshTasksEXT`, `vkCreatePipelineLayout`, `vkCreateDescriptorSetLayout`. The debug
  ranges are how dispatch-level structure appears.
- `vkCmdPipelineBarrier2` is recorded only when it directly follows another barrier call in the
  same command buffer (the first barrier after any other call is dropped). In the scan test, 192
  barriers were recorded as 48. Barrier counts in `vulkan_api_sum` are a **lower bound**.
- GPU workload rows need **nsys 2026.3.1 or newer**; 2026.2 silently records none.
- `--vulkan-gpu-workload=batch` produces a single coarse row with no label text, so
  `vulkan_gpu_marker_sum` returns "no data". Keep `individual`.
- On compute and transfer queues, command buffer execution ranges may appear to end earlier than
  they actually do.
- No CLI report reads the raw `VULKAN_WORKLOAD` rows; use SQLite or the GUI.
- On Linux the target must be launched under `nsys`; attaching is not supported.
- Root is not required for Vulkan tracing.
