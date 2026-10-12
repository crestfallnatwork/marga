# CUDA ablations of MARGA's triangle-counting pipeline

A hand-written CUDA implementation of MARGA's TC pruning (orientation cull +
bounded lists + smallest-list intersection) with plain one-thread-per-edge
scheduling. It measures how far the pruning alone gets on the same graphs, to
attribute MARGA's advantage to its mesh-shader execution model rather than to
the algorithm alone.

## Arm

One thread per edge, no task/mesh decomposition. `--cull` selects the pruning
level:

- `none` — all symmetrized edges, full-list intersection (each triangle counted 6x)
- `orient` — u<v edges only, full-list intersection (3x)
- `bounded` — u<v edges + bounded lists, counted once (MARGA's algorithm)

`--preorient` builds the oriented (u<v) edge list in a two-pass device pre-pass
(per-vertex count + CUB prefix sum, then write) and runs the arm on it. The
pre-pass is timed separately (`prepass_ms`) and preserves edge order (no load
balancing).

## Benchmark protocol

- `--warmup N` untimed runs then `--reps N` timed runs; CUDA events around the
  whole run (host round-trips included), averaged.
- `--batch N` edges per launch (default 2^24, MARGA's TC batch size, one host
  round-trip per batch).
- `--validate` compares the triangle count against the CPU reference.
- Run one GPU benchmark at a time.

## Build

```bash
cmake -S ablations/cuda -B <build-dir> -DCMAKE_BUILD_TYPE=Release
cmake --build <build-dir> --parallel
```

`-DMARGA_CUDA_ARCH=<cc>` overrides the target compute capability (nvcc
otherwise defaults to sm_75).

## Example

```bash
tc_ablate --graph <graph.mtx> --cull bounded --reps 3 --warmup 1 --validate
tc_ablate --graph <graph.mtx> --cull bounded --preorient --reps 3 --warmup 1
```

The graph is loaded symmetrized with sorted, deduplicated adjacency (MARGA's TC
import semantics) and cached next to the `.mtx` as `<name>.mtx.sym.csrbin`
(use `--no-cache` to skip).
