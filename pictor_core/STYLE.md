# pictor_core style

Elegant, minimal code. Every abstraction has to pay rent: you pay for what you use in both cost
and complexity. The API should be clean enough that a GPGPU workload is a Slang kernel, a few
lines of host code, and nothing else - Vulkan boilerplate lives behind the library and the
shader codegen.

This document is the semantic map of the project plus the rules we hold ourselves to. It is
normative for new code; existing deviations are listed at the end with their disposition.

## 1. What this library is

pictor_core is a Vulkan 1.3 runtime for GPGPU work in all its kinds: compute dispatch,
mesh-shader amplification, and headless rasterization used as another compute vehicle. It is not
a game engine and not a general renderer. The focus is:

- data-parallel primitives (scan and friends) with explicit memory control;
- hardware-conscious execution (BDA buffers, subgroup intrinsics, decoupled look-back);
- codegen that removes the Vulkan boilerplate (pipelines, push constants, reflection).

## 2. Semantic map

### Layers

| Area | Public surface | Responsibility |
|---|---|---|
| `memory/` | `Allocator`, `PoolAllocator`, `PinnedMemoryAllocator`, `HeapAllocator`, `AllocationInfo`, `AllocationType` | Raw GPU/host memory. Allocators own VMA-backed buffers and BDA addresses. |
| `execution/` | `ShaderCompiler`, `Stream`, `ComputeExecutor`, `MeshExecutor`, `BaseExecutor`, `GpuTimer`, `Access`, `ResourceBarrier` | Command recording, Slang compilation, pipelines, GPU phase timing, synchronization vocabulary. |
| `context/` | `Context`, `DeviceInfo`, `Expected`, `Optional` | Instance/device/queue setup, device facts and limits, feature and extension enables, allocator and stream ownership, transfer/barrier helpers. |
| `algorithms/` | `Scan` (+ future join primitives) | Multi-pass GPU algorithms built only on the public API. |
| codegen | `pictor_add_shader`, `pictor_reflect` | Turns a `.slang` file into `<name>.slang.h` (embedded source closure + search paths) and `<name>.shader.h` (structs, traits, executor aliases). |

### How a workload flows

1. `Context::create()` selects a device (discrete preferred, requires mesh shaders and int64),
   enables the extension/feature set, sets up VMA, the root allocators, the Slang compiler and
   the default stream.
2. Buffers come from the context's allocators (`ctx->pool_allocator` for device,
   `ctx->pinned_allocator` for host staging; the v2 API adds `allocate_device` / `allocate_host`
   / `allocate_image` and keeps the pointers for compatibility).
3. An executor is created from a generated trait (`Scan32ComputeExecutor::create(ctx)`), which
   compiles the embedded Slang source for the device's subgroup size and builds the pipeline.
4. `executor->record(...)` records the dispatch (and its push constants / push descriptors) into
   the current stream command buffer. Record paths are allocation-free.
5. `ctx->fsync()` flushes and waits; pinned memory is read back after `ctx->invalidate`.
6. `ctx->deallocate(info)` dispatches to the allocator recorded in the handle.

### Patterns

- **Factories**: `static Expected<std::unique_ptr<T>, vk::Result> create(...)` with private
  constructors. `std::make_unique` cannot call a private constructor, so `new T(...)` inside
  `create` is deliberate - not an inconsistency.
- **Handles**: `AllocationInfo` is a value struct (buffer, allocation, mapped pointer,
  size/offset, BDA address, owning allocator). Never construct one by hand.
- **Error model**: `Expected<T, vk::Result>` / `Optional<T>` and `vk::Result` returns; no
  exceptions (`VULKAN_HPP_NO_EXCEPTIONS`).
- **Executors**: header-only templates parameterized by a generated `TCode` (source, entry
  points, workgroup shape), `TParams` (push constants) and an optional `TResources`
  (push-descriptor bindings). `record` passes a stack params struct to a static callback.
- **Pimpl** where implementation weight justifies it (`ShaderCompiler`, `Scan`).
- **Public members** on `Context` (device handles, allocator pointers, subgroup size, cached
  device info) are deliberate: the context is the service locator for the library.
- **GPU timing**: `GpuTimer` records named RAII laps on one stream. Read results after `fsync`, never
  with an open lap, and do not submit in the middle of a lap.

## 3. Rules

### 3.1 Formatting

`clang-format` with the checked-in `.clang-format` is authoritative: 4-space indent, 100
columns, braces after function definitions, K&R control statements, left-aligned pointers and
references (`Type* p`, `Type& r`), consecutive assignments aligned, case-sensitive include
sorting within preserved blocks.

- `scripts/format.sh` formats the whole tree or only the staged changes.
- The pre-commit hook runs it automatically (see Git).
- Do not fight the formatter; if the output is ugly, restructure the code.

### 3.2 Files and includes

- One SPDX header per file: `/* SPDX-License-Identifier: MPL-2.0 */`. The legacy `Author:`
  line may remain on existing files; new files do not need it.
- `#pragma once`, not include guards.
- Project headers are quoted: `#include "pictor_core/execution/stream.hpp"`. Generated headers
  are quoted (`#include "scan.shader.h"`). Third-party and standard headers use angle brackets.
- Include order: project, generated, third-party, standard library - one block each. Include
  what you use; never rely on a transitive include.

### 3.3 Naming

| Entity | Convention | Example |
|---|---|---|
| Types, enums, concepts | PascalCase | `ComputeExecutor`, `ScanKind` |
| Methods, free functions | camelCase | `create_stream`, `get_vulkan_access_info` |
| Private members | `_camelCase` | `_impl`, `_recording` |
| Public constants | `c` prefix | `cDefaultStream` |
| Local/file constants | `k` prefix | `kScanItems32` |
| New enum values | `eName` | `AllocationType::eBuffer` |
| Files | `snake_case.hpp` / `.cpp` | `heap_allocator.hpp` |
| Shader entry points | `<group>_<stage>` | `scan32_compute`, `add_mesh` |

Existing enums with bare values (`Access::Read`, `ShaderStage::Compute`,
`ScanKind::Inclusive`) are kept for compatibility; do not mix styles in new code.

### 3.4 API design

- **Extension over change.** Public names and shapes are depended on by downstream projects;
  extend interfaces, never rename or reshape without a strong, stated justification.
- **Minimum surface.** No wrapper type that only forwards; no abstraction without a second user
  or a measurable benefit.
- **Context owns resources.** Callers never reach into allocator objects; memory requests go
  through `Context`. The legacy `pool_allocator` / `pinned_allocator` pointers remain for
  compatibility and will not be extended.
- **Pay for what you use.** Opt-in everywhere: no descriptor set unless a shader declares
  bindings, no clear unless asked, no sampler work unless sampled, no cost for unused stages.
- **Prefer compile-time facts.** When codegen knows something (attachment count, stage
  presence, workgroup shape), expose it as a constant or template parameter instead of runtime
  state.
- **No heap on record paths.** Fixed-size stack arrays and caller-owned buffers only.
- **No hidden commands.** `record()` records exactly the draw/dispatch it is asked for; layout
  transitions and barriers are explicit `Context` calls.

### 3.5 Errors and invariants

- No exceptions. Check every Vulkan call; return `Expected` / `vk::Result`. Use `(void)` for
  deliberately ignored results (destructors).
- `assert` for programmer errors only, never for runtime conditions.
- Every device-dependent quantity (subgroup size, limits, memory budgets) is queried at runtime;
  never hardcode it.

### 3.6 Memory

- Device buffers are `AllocationInfo` handles and expose BDA addresses; they are the shader-side
  interface (no descriptor sets for buffers).
- Host staging is pinned and mapped: `flush` after writes, `invalidate` before reads.
- Deallocate through `ctx->deallocate` so the owning allocator is used.
- Barriers use the `Access` vocabulary; do not spell raw stage/access masks in callers.

### 3.7 Executors and shaders

- Executors are header-only templates. `create(ctx)` compiles entries and builds the pipeline;
  `record(...)` is allocation-free and compile-time-dispatched (`if constexpr` on `TCode`).
- Slang sources are the single source of truth for parameter structs, bindings and workgroup
  shape; host code uses the generated structs and traits. Workgroup sizes are reflected at
  runtime, not generated.
- Push constants carry BDA pointers and scalars; add explicit padding where layout demands it.
- `WAVE_SIZE` is injected at compile time from the device subgroup size; never hardcode it.
- Shader options (`CompileOptions`) are preprocessor defines only. `subgroup_size` stays a
  separate parameter, `WAVE_SIZE` is never one of them, and a supplied `WAVE_SIZE` define is
  rejected with a diagnostic.
- Validate all graphics/executor work with the Khronos validation layers.

### 3.8 Codegen

- Generated headers are build artifacts: never hand-edit; change the `.slang` source or the
  generator.
- Host/device constants that must agree (for example `kScanItems32` with `VPT32`) are
  documented next to one of the two definitions.
- Shader headers are generated by `pictor_reflect` from Slang's reflection API. The only
  convention it parses is the `PICTOR_`-prefixed guarded literal default (in live,
  comment-stripped code) that declares a shader tunable (see `docs/shader-codegen.md`);
  everything else is reflected, not parsed.
- Push constants are resolved per entry; a module may declare several distinct layouts. Do not
  add annotations for this: `pictor_reflect` uses the entry's linked SPIR-V when a module has more
  than one layout, and a task/mesh pair must resolve to the same layout.
- Shaders may `import` device-side modules and `#include` files; `pictor_add_shader(... INCLUDE_DIRS
  <dir>...)` passes the search paths. The generator embeds the whole dependency closure and the
  runtime compiles it through an in-memory file system, so imports resolve identically at build
  time and at run time, without file-system access.

### 3.9 Tests

- Top-level builds wire `ctest`: `library_test` (allocators) and `scan` (GPU correctness against
  a CPU reference).
- New algorithms ship with a correctness test and a size sweep covering the boundary cases
  (0, 1, tile size +/- 1, large).
- Tests may use raw allocator access today; migrate them to the v2 Context allocation API as it
  lands.

### 3.10 Git

- Focused, descriptive commits; do not push without permission.
- Hooks are checked in under `.githooks/`; enable them once per clone:
  `git config core.hooksPath .githooks`.
- The pre-commit hook formats staged C++ files (`git-clang-format`) and runs `clang-tidy` on
  them when `build/compile_commands.json` exists. `bugprone-*` and `performance-*` findings are
  errors; `modernize-*` and `readability-*` are advisory warnings (see `.clang-tidy`).

## 4. Known inconsistencies

### Fixed with this document

- clang-format drift across 12 files; `.clang-format` `Standard: Latest` -> `c++20`.
- Angle-bracket project includes in `algorithms/` -> quoted, matching the rest of the tree.
- Empty dead files `include/pictor_core/memory/vulkan_allocator.hpp` and
  `include/pictor_core/memory/pool_allocator_test.cpp` removed.
- Lint baseline added (`.clang-tidy`) and the tree cleaned to pass it: C-style casts ->
  `static_cast`, redundant `virtual`/`inline`/`typename`, narrowing conversions, missing
  `[[nodiscard]]`, SFINAE -> C++20 `requires`, C arrays -> `std::array`, integer-to-pointer
  arithmetic -> typed pointer arithmetic, `std::endl` -> `'\n'`.
- Generated traits: `static inline constexpr` -> `static constexpr`.
- `Context::get_device_free_memory` / `get_host_free_memory` made `const` (non-breaking);
  `Context::fill` no longer returns the result of a void call.
- Include what you use: `<cassert>` and `<limits>` added to `context.cpp`.
- CMake: trailing whitespace, missing `CMAKE_EXPORT_COMPILE_COMMANDS`, the scan test not wired
  into CTest; `library_test` is now a registered test too.
- Tooling: `scripts/format.sh`, `scripts/lint.sh`, and a checked-in pre-commit hook
  (`.githooks/pre-commit`).

### Open, non-API

- `test/shaders/{bfs,library_test}.slang` are unreferenced examples (never compiled). Decide
  between an `examples/` directory and wiring them into tests.
- `test/test.cpp` is an allocator timing demo registered as `library_test`; consider renaming it
  when tests grow.
- `ScanKind::Inclusive/Exclusive` and the `kScan*` constants predate this document; align new
  code with section 3.3 instead.

### API-deferred (do not fix casually)

- `Context::fill` has two overloads with swapped parameter order (`(dst, value, offset, size)`
  and `(dst, offset, size, value)`).
- `AllocationInfo::operator T*` is an implicit conversion from the BDA address.
- `Access` maps only compute/mesh stages; the v2 design extends it with raster stages and image
  accesses.
- `Allocator` has no image support; the v2 design adds non-pure `allocate_image` /
  `deallocate_image`.
- `HeapAllocator::create` takes raw `Context*` / `Allocator*` where other factories take
  references.
