# Benchmarks

Comparison harnesses for the `algorithms/` primitives. These are **not tests**: they are not wired
into `ctest`. Configure with `-DPICTOR_BUILD_BENCHMARKS=ON` and drive them with `scripts/bench.sh`.

```bash
scripts/bench.sh                       # every benchmark with default sizes
scripts/bench.sh scan                  # scan: pictor::Scan vs cub::DeviceScan
scripts/bench.sh scan-vulkan --sizes 1048576,4194304,16777216
scripts/bench.sh scan-cuda   --bits 64 --kinds inclusive
scripts/bench.sh --build               # configure + build only
```

The Vulkan harness links `pictor_algorithms`; the CUB harness is the reference and is built only
when `CUDAToolkit` is found (CUB ships with the toolkit, or comes from the `gpujoin` env's
`rapids`/`cccl` include tree).

## Rules

- Time GPU work with GPU timestamps (Vulkan) / CUDA events (CUDA); never host clocks around async
  work.
- Warm up before timing; report the median.
- Pin clocks on the benchmark server and keep the GPU otherwise idle (see the `scratch/` notes);
  record the commit hash and GPU state alongside any numbers.
- Use the same synthetic generator (`bench/primitives/synthetic.hpp`) and seed for every variant.

## Scan (`pictor_bench_scan_vulkan`, `pictor_bench_scan_cuda`)

Head-to-head of `pictor::Scan` (single-pass decoupled look-back) against `cub::DeviceScan` on the
same synthetic sequence and sizes; both verify every configuration against the streaming host
reference.

```bash
./build/bench/primitives/pictor_bench_scan_vulkan --bits 32,64 --kinds exclusive,inclusive
./build/bench/primitives/pictor_bench_scan_cuda   --bits 32,64 --kinds exclusive,inclusive
```

Options: `--bits 32,64`, `--kinds exclusive,inclusive`, `--sizes n,n,...`, `--iters N`,
`--warmup N`, `--no-verify`.

## Sort (`pictor_bench_sort_vulkan`, `pictor_bench_sort_cuda`)

Head-to-head of `pictor::RadixSort` (Onesweep) against `cub::DeviceRadixSort` on the same synthetic
sequence and sizes; both verify the sorted order (the twiddled key, masked to the range) and, for
key-value sorts, the exact key-payload pairing.

```bash
./build/bench/primitives/pictor_bench_sort_vulkan --keys 64 --order signed --sizes 4194304
./build/bench/primitives/pictor_bench_sort_cuda   --type i64 --sizes 4194304
./build/bench/primitives/pictor_bench_sort_vulkan --values 64 --descending
./build/bench/primitives/pictor_bench_sort_cuda   --type u32 --values 64 --descending
```

The Vulkan options mirror `SortSpec`: `--keys 32|64`, `--order unsigned|signed|float`,
`--begin-bit N`, `--end-bit N`, `--descending`, `--values 0|32|64`, plus `--shape WAVES,VPT`
(pin the workgroup shape), `--sizes`, `--iters`, `--warmup`, `--no-verify`. The CUB side takes
`--type u32|i32|f32|u64|i64|f64` instead of `--keys`/`--order`, and the same remaining flags.
