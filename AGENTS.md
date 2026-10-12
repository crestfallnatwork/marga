# AGENTS.md — working notes for AI agents

MARGA (*Raster has the Edge: Graphics Pipelines for Graph Queries*) is a graph
processor built on the mesh shader pipeline (MSP). It maps graph algorithms —
breadth-first search (BFS), triangle counting (TC), and PageRank — onto the
task + mesh stages of the rasterization pipeline instead of classic compute
shaders.

## Current work

Active work items, findings and handoff state live in `scratch/WORKLOG.md`
(untracked). Machine/environment notes live in `scratch/ENVIRONMENT.md`.

## Repository layout

- `src/*.cpp` — host-side drivers (`bfs`, `pagerank`, `tc`), the graph
  importer (`csr`), and CPU validators (`*_validation`).
- `src/shaders/*.slang` — GPU task/mesh shaders, compiled to SPIR-V at build
  time by `pictor_core`.
- `pictor_core/` — Vulkan memory/execution layer (`Context`, allocators,
  `MeshExecutor`/`ComputeExecutor`, Slang integration).
- `baselines/` — comparison systems (cuGraph, GunRock, GraphRTX).
- `artifacts/` — benchmark CSVs and plots.
- `ablations/cuda/` — standalone CUDA ablation harness (see its README);
  builds separately from the main project.
- `scratch/` — untracked local notes, scripts and logs.

## Style

- C++20; prefer `std::format` over iostream formatting.
- Comments describe what the code does. Measurements, expectations, and
  explanations of why something is slow belong in notes/paper, not in code.
- No machine-specific or personal details in committed files (hostnames,
  paths, specific test hardware).
- Keep kernels self-contained and named after the variant they implement.
- Performance numbers: warm up once, time on the GPU, report the average of
  ≥5 runs, and record the GPU/driver that produced them.
- Prefer `scratch/` for logs and CSV outputs; copy summaries to `artifacts/`
  only when they are worth keeping.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

The first configure fetches pinned dependencies (Vulkan-Headers, volk,
VMA-Hpp, Slang), so it needs network access.

## Workflow conventions

- Isolate experiments on branches; `main` is the integration branch.
- Commit small and often.
- The CUDA ablation harness on `main` is restricted to the published TC arms
  (static, task/mesh, orientation pre-pass) — keep new experiment arms on
  separate branches and merge back deliberately.
