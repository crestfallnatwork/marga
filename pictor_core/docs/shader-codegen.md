# Shader codegen and the shader pipeline

`pictor_core` compiles Slang at run time, but the host/device interface is generated at build
time. This guide covers the whole path: how a `.slang` file becomes generated C++ headers, how an
executor turns those headers into a pipeline, and the rules that make the two halves line up.

Runnable examples: `examples/compute` (minimal compute), `examples/mesh` (task + mesh),
`examples/modules` (device-side modules). Each is a self-verifying `ctest` target.

## The big picture

Two stages, deliberately split:

```
build time                                         run time
┌────────────────────────────┐                     ┌──────────────────────────────────┐
│ .slang + imported modules  │                     │ generated headers                │
│           │                │                     │           │                      │
│           ▼                │   <stem>.slang.h    │           ▼                      │
│     pictor_reflect         │ ──────────────────▶ │  Executor::create(ctx)           │
│  (Slang reflection API)    │   <stem>.shader.h   │  ShaderCompiler JIT              │
│           │                │                     │  WAVE_SIZE = device subgroup size│
│           ▼                │                     │           │                      │
│  embedded source closure   │                     │           ▼                      │
│  struct layouts, entries   │                     │  SPIR-V → pipeline → executor    │
└────────────────────────────┘                     └──────────────────────────────────┘
```

- **Build time** runs `pictor_reflect`. It loads the shader with Slang's reflection API, collects
  the whole dependency closure (`import`ed modules and `#include`d files), and emits two headers.
  It also scans the closure for `PICTOR_`-prefixed guarded defaults in live code (comments and
  inactive `#if` branches are ignored) and emits them as tunables
  (see [Shader defines and tunables](#shader-defines-and-tunables)). It does **not** produce
  runtime SPIR-V and does not run `slangc`; when a module declares more than one distinct
  push-constant layout (or another closure file mentions `push_constant`), it compiles each entry
  once to see which layout that entry uses (see [Push constants](#push-constants)), and uses the
  result only to choose the generated `TParams`; the runtime still compiles the shader from
  source.
- **Run time** compiles the embedded sources with Slang's JIT, targeting the actual device: the
  `WAVE_SIZE` macro is defined as the device's subgroup size, and the workgroup size is reflected
  from the compiled program. The SPIR-V goes straight into a Vulkan pipeline.

Why the split? The subgroup size is a property of the device, not of the build machine. A shader
that uses `WAVE_SIZE` (or sizes its `[numthreads]` with it) must be compiled for the GPU it will
run on; a build-time SPIR-V would bake in a guess. Keeping runtime SPIR-V out of the generated
headers also means they are portable — the same headers work on any device, and only the run-time
compile changes. The cost is a one-time compile per executor at `create()` (tens of milliseconds;
see [Performance](#performance)).

### The generated files

For a shader `add.slang`:

| File | Contents |
|---|---|
| `add.slang.h` | `add_shader_sources`: the embedded source closure (`cFiles`), the virtual search paths (`cSearchPaths`), and the `ShaderSources` value (`cSet`). |
| `add.shader.h` | `add_shader`: the push-constant struct(s), entry traits (`entry_point` / `task_entry` / `mesh_entry` and `sources`), executor aliases, and — when the shader declares tunables — the `<Prefix>Tunables` struct, its `defines()`, and each trait's `using Tunables = ...` alias. |

Both are written into the calling `CMakeLists.txt`'s binary directory. They are build artifacts:
never edit them by hand — change the `.slang` source or `pictor_reflect`.

## Getting started

The minimal compute shader (`examples/compute/add.slang`):

```slang
struct AddParams {
    uint64_t a_addr;
    uint64_t b_addr;
    uint64_t out_addr;
    uint32_t count;
};

[[vk::push_constant]]
uniform AddParams p;

[shader("compute")]
[numthreads(64, 1, 1)]
void add_compute(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x < p.count)
        ((float*)p.out_addr)[tid.x] = ((float*)p.a_addr)[tid.x] + ((float*)p.b_addr)[tid.x];
}
```

`CMakeLists.txt`:

```cmake
pictor_add_shader(example_compute_shader add.slang)

add_executable(example_compute add.cpp)
target_link_libraries(example_compute
    PRIVATE
        pictor_core
        example_compute_shader
)
add_test(NAME example_compute COMMAND example_compute)
```

The host program (`examples/compute/add.cpp`, abbreviated):

```cpp
#include "pictor_core/context/context.hpp"

#include "add.shader.h" // generated

using namespace pictor;

auto ctx_res = Context::create();
if (!ctx_res.ok())
    return 1;
auto ctx = std::move(ctx_res.val());

auto exec_res = add_shader::AddComputeExecutor::create(*ctx);
if (!exec_res.ok())
    return 1;
auto exec = std::move(exec_res.val());

// Allocate, upload and barrier the inputs (see the full example for the boilerplate).
auto d_a   = ctx->pool_allocator->allocate(bytes, alignof(float));
auto h_out = ctx->pinned_allocator->allocate(bytes, alignof(float));

add_shader::AddParams params{};
params.a_addr = d_a.address;
params.count  = count;

exec->record(count, 1, 1, params); // ceil(count / 64) workgroups

ctx->barrier(d_out, Access::Write, Access::TransferRead);
ctx->copy(d_out, h_out);
ctx->fsync();
ctx->invalidate(h_out);
// verify, deallocate, destroy executors before the Context
```

That is the whole contract: the shader declares its interface, `pictor_reflect` mirrors it into
C++, and the executor compiles and dispatches it. No annotations, no descriptor set plumbing, no
hardcoded workgroup size on the host.

## Entry points and generated names

Every `[shader(...)]` entry point in the module is reflected. The entry name groups entries into a
family: the **prefix** is everything before the last `_`, and each family produces one trait.

| Entry points | Generated trait | Generated executor alias |
|---|---|---|
| `<prefix>_compute` | `<Prefix>Compute` | `<Prefix>ComputeExecutor` |
| `<prefix>_task` + `<prefix>_mesh` | `<Prefix>Mesh` | `<UPPER_PREFIX>Executor` |

`<Prefix>` capitalizes each underscore-separated word of the prefix (first letter upper, rest
lower): `scan32` → `Scan32`, `init` → `Init`. The mesh alias uppercases the raw prefix:
`simple` → `SIMPLEExecutor`, `tile` → `TILEExecutor`, `add` → `ADDExecutor`.

Examples from the tree:

| Shader entries | Generated C++ |
|---|---|
| `add_compute` | `add_shader::AddCompute`, `add_shader::AddComputeExecutor` |
| `tile_task` + `tile_mesh` | `tile_shader::TileMesh`, `tile_shader::TILEExecutor` |
| `scan32_compute` | `scan_shader::Scan32Compute`, `scan_shader::Scan32ComputeExecutor` |
| `init_compute` | `scan_shader::InitCompute`, `scan_shader::InitComputeExecutor` |

Notes and edge cases:

- The namespace is `<stem>_shader`, where `<stem>` is the shader file name without its extension
  (`add.slang` → `add_shader`).
- A name without `_` is its own prefix: `main` → `MainCompute`.
- Compute and mesh families are independent: one file may contain `add_compute` and
  `tile_task`/`tile_mesh`, producing both traits.
- Vertex and fragment entries are reflected (visible in `--dump`) but no C++ is emitted for them
  yet — rasterization is in progress.
- A `_task` without a matching `_mesh` (or vice versa) produces no trait. If several entries share
  a prefix and stage, only the first one in the module is emitted.
- Trait names are case-folded: avoid prefixes that differ only in case (`Add` and `add` would both
  become `Add`).

## Push constants

The push-constant block is the host/device interface. Declare a struct and mark a uniform of that
type with `[[vk::push_constant]]`; nothing else is required:

```slang
struct AddParams {
    uint64_t a_addr;
    uint64_t b_addr;
    uint64_t out_addr;
    uint32_t count;
};

[[vk::push_constant]]
uniform AddParams p;
```

`pictor_reflect` emits a C++ struct with the same name and fields. Offsets are taken from Slang's
uniform layout; whenever a gap exists between fields the generator emits an explicit byte-array
pad, so the C++ member offsets equal the shader's:

```cpp
struct AddParams {
    uint64_t a_addr;
    uint64_t b_addr;
    uint64_t out_addr;
    uint32_t count;
};
```

`record(...)` passes `sizeof(Params)` bytes at push-constant offset 0. The unit of resolution is
the **block layout** (the struct type), not the variable name. When a module declares more than
one distinct layout, `pictor_reflect` compiles the entry's linked SPIR-V once to see which layout
it uses, and the generated executor alias binds that layout. Multiple blocks of the same type are
one layout and are accepted. A task + mesh pair must resolve to the same layout (or one stage may
use none); a mismatch is a generation error. A module with a single layout and no `push_constant`
mention outside the entry file skips the compile and binds that layout to every entry (an entry
that does not use it simply ignores the data). Entries may share a layout; several compute passes
can bind the same block.

For example, two compute entries with different blocks generate two correctly typed executors:

```slang
[[vk::push_constant]] uniform BuildIn build;
[[vk::push_constant]] uniform ProbeIn probe;

[shader("compute")] void build_compute(...) { ... build ... }
[shader("compute")] void probe_compute(...) { ... probe ... }
```

```cpp
using BuildComputeExecutor = pictor::ComputeExecutor<BuildCompute, BuildIn>;
using ProbeComputeExecutor = pictor::ComputeExecutor<ProbeCompute, ProbeIn>;
```

### Field kinds

| Slang field | Generated C++ | Notes |
|---|---|---|
| `uint32_t`, `float`, `bool`, ... | the matching scalar | `int8_t` ... `int64_t`, `uint8_t` ... `uint64_t`, `float`, `double` |
| `float3 v` | `float v[3]` | vectors are emitted as scalar arrays |
| `float4 w[2]` | `float w[2][4]` | array dimensions outermost first |
| `struct Inner inner` | `Inner inner` | nested structs are emitted recursively, once each |
| `float* p` | `float* p` | BDA pointer (8 bytes) |
| enum | the underlying scalar type | `enum Mode : uint32_t` → `uint32_t` |
| `float4x4 m` | `uint8_t m[64]` | anything unmappable falls back to a byte array of the reflected size |
| `uint mode : 8` | `$bit_field_backing_0` | Slang-synthesized name; avoid bitfields |

Pointers: scalar pointers (`float*`, `uint32_t*`) map cleanly. A pointer to a vector
(`float4*`) is currently emitted as an array of scalar pointers, which does not match the 8-byte
Slang field — use scalar pointers, or pass a `uint64_t` address and cast in the shader as
`scan.slang` does.

### Layout caveats

- **No trailing padding.** Only gaps *between* fields get `_pad_` arrays; the padding after the
  last field is not emitted, so `sizeof(TParams)` can be smaller than the shader's block. The
  accessed fields are still at the right offsets.
- **Vectors keep offsets, not alignment.** A single `float3` is followed by a `_pad_` array before
  the next field (correct), but an *array* of three-component vectors uses a 16-byte stride in
  Slang (`float3 v[2]` occupies 32 bytes) while the generated C++ array has a 12-byte stride.
  Subsequent fields then land at the wrong offset. Keep push-constant arrays to scalars or
  `float4`/`float2`, or pass an address instead.
- **Bitfields** produce Slang's synthesized backing-field name (`$bit_field_backing_0`). GCC and
  Clang accept `$` in identifiers as an extension; it is not portable and the field is not usable
  by the name you wrote. Avoid bitfields in push constants.
- **Empty or missing push-constant blocks.** A shader with no `[[vk::push_constant]]` (or an empty
  struct, which Slang elides) generates an executor alias with `void` params (for example
  `ComputeExecutor<Code, void>`), which fails to compile when used (`forming reference to void`).
  Always declare at least one field. See [Troubleshooting](#troubleshooting).

The push-constant interface comes from Slang's reflection data; annotations on it are not parsed
and no `@bind`-style annotation is used. Resolution is automatic: when a module declares more than
one distinct layout (or another closure file mentions `push_constant`), the entry's linked SPIR-V
identifies the layout it references. Push constants must be declared in the entry
file (or a `#include`d file); a block declared in an `import`ed module is not visible in the
program layout, and an entry that uses one is rejected unless the entry file declares a block with
the same layout. The resolution compile uses the reflection `WAVE_SIZE` (default 32, or
`--wave-size`) and the shader's default tunables, matching the rest of the reflection pass; do not
make push-constant usage or layout depend on `WAVE_SIZE` or a device-specific tunable. The one source convention the generator parses is the tunable
declaration (see [Shader defines and tunables](#shader-defines-and-tunables)).

## Generated headers

### `<stem>.slang.h` — the embedded sources

```cpp
#pragma once
#include "pictor_core/execution/shader_sources.hpp"

#include <array>
#include <string_view>

namespace scan_shader_sources {

inline constexpr std::array<std::string_view, 1> cSearchPaths{"/sp0"};

inline constexpr std::array<pictor::ShaderSourceFile, 1> cFiles{{
    {"/sp0/scan.slang", R"pictor(
... the full source text ...
)pictor"},
}};

inline constexpr pictor::ShaderSources cSet{"/sp0/scan.slang", cSearchPaths, cFiles};

} // namespace scan_shader_sources
```

- `cFiles` is the entire dependency closure, verbatim, under virtual paths.
- `cSearchPaths` mirrors the build-time search paths: `/sp0` is the shader's own directory,
  `/sp1`... are the `INCLUDE_DIRS` entries in order.
- `cSet` is the `pictor::ShaderSources` value (`entryPath`, `searchPaths`, `files`) that
  `ShaderCompiler` consumes. The runtime mounts it as an in-memory file system and loads
  `entryPath` from it.
- The raw-string delimiter is chosen so it cannot appear in any embedded source.

### `<stem>.shader.h` — the C++ interface

```cpp
#pragma once
#include "scan.slang.h"
#include "pictor_core/execution/compute_executor.hpp"
#include "pictor_core/execution/mesh_executor.hpp"

namespace scan_shader {

struct ScanParams {
    uint64_t input;
    ...
};

struct Scan32Compute {
    static constexpr std::string_view entry_point = "scan32_compute";
    static constexpr const pictor::ShaderSources& sources = scan_shader_sources::cSet;
};

using Scan32ComputeExecutor = pictor::ComputeExecutor<Scan32Compute, ScanParams>;

} // namespace scan_shader
```

The trait members:

- Compute: `entry_point` (the Slang entry name) and `sources`.
- Mesh: `task_entry`, `mesh_entry` and `sources`.
- The executor aliases bind the trait and the push-constant struct to the header-only
  `ComputeExecutor` / `MeshExecutor` templates. `create(ctx)` compiles the trait's entry points
  from `sources` and builds the pipeline.

You normally only touch the alias, the params struct, and (for manual dispatch arithmetic) the
workgroup-size members described below.

## Workgroup sizes and dispatch

`[numthreads(...)]` lives in the shader and may use `WAVE_SIZE`. At executor creation the compile
links the program and reads the workgroup size back from the compiled shader
(`CompileResult::threadGroupSize`), then passes it to the pipeline. The host never declares it.

The executor exposes the reflected sizes as public constants:

- `ComputeExecutor`: `wgX`, `wgY`, `wgZ`.
- `MeshExecutor`: `taskWgX/Y/Z` and `meshWgX/Y/Z`.

Dispatch helpers:

| Call | Meaning |
|---|---|
| `record(x, y, z, params)` | Work-item counts: dispatches `ceil(x / wgX)`, `ceil(y / wgY)`, `ceil(z / wgZ)`. Compute divides by the compute workgroup size; mesh divides by the **task** workgroup size (the task stage decides how many mesh groups to launch). |
| `record_raw(x, y, z, params)` | Mesh only: `x/y/z` are task-workgroup counts passed straight to `vkCmdDrawMeshTasksEXT`. |
| `record_indirect(allocation, params)` | Reads the dispatch from a device buffer: `vkCmdDispatchIndirect` (compute) or `vkCmdDrawMeshTasksIndirectEXT` (mesh, one command, stride 0). The buffer needs indirect usage (the pool allocator's buffers have it) and an `Access::IndirectRead` barrier. |

Because the size is reflected from the compiled shader, changing `[numthreads]` requires no host
changes, whether the new value comes from an edit or from a tuner overriding a `PICTOR_*` macro
used in `[numthreads]`. Never hardcode a subgroup size: `ctx.subgroup_size` is the device's value
and is what `WAVE_SIZE` is defined to at run time. The mesh task stage and mesh stage may have
different workgroup sizes; both are reflected independently.

## Shader defines and tunables

Some shader parameters have to be known at compile time — they size an array, appear in
`[numthreads]`, or select a code path. Two pieces cover them: `CompileOptions` passes
preprocessor defines into the JIT compile, and shader-declared tunables let the shader advertise
those defines (with defaults) so `pictor_reflect` can mirror them into a generated C++ struct.

### `CompileOptions` and `ShaderDefine`

`include/pictor_core/execution/shader_defines.hpp`:

```cpp
struct ShaderDefine {
    std::string_view name;
    std::string_view value;
};

struct CompileOptions {
    std::span<const ShaderDefine> defines{};
};
```

`compile(...)` and both executor factories take `CompileOptions` as the last argument:

```cpp
CompileResult compile(const ShaderSources& sources, const std::string& entryPoint,
                      ShaderStage stage, uint32_t subgroupSize = 32,
                      CompileOptions options = {});

ComputeExecutor<Code, Params>::create(ctx, specInfo = nullptr, options = {});
MeshExecutor<Code, Params>::create(ctx, specInfo = nullptr, options = {});

// tuned variant: the injected callable picks the values (see "Injecting a tuner at executor
// construction")
ComputeExecutor<Code, Params>::create(ctx, tuner, specInfo = nullptr);
MeshExecutor<Code, Params>::create(ctx, tuner, specInfo = nullptr);
```

The defines are views, so static ones cost nothing:

```cpp
static constexpr pictor::ShaderDefine kDefines[] = {{"PICTOR_WAVE_MATCH", "1"}};
auto exec = MyExecutor::create(ctx, nullptr, pictor::CompileOptions{kDefines});
```

`subgroup_size` stays a **separate parameter**, and `WAVE_SIZE` must **not** appear in the
defines list: the subgroup size is a property of the device, and the runtime always injects
`WAVE_SIZE` as the first preprocessor macro of every compile. `compile` rejects a define named
`WAVE_SIZE` by returning a failed `CompileResult` whose `diagnostics` names it, rather than
letting Slang silently keep the runtime's value. `CompileOptions` carries only shader options.

### The `PICTOR_`-prefixed guarded-default convention

A shader declares a tunable with a guarded literal default:

```slang
#ifndef PICTOR_VPT
#define PICTOR_VPT 18
#endif
```

- **The prefix is the marker.** Only `PICTOR_`-prefixed names are tunables; this keeps
  runtime-injected facts (`WAVE_SIZE`) and shader-internal helpers out of the set.
- **The guarded literal default is the declaration** — there is no annotation and no separate
  list. The guard is not extra boilerplate: the build-time reflection pass compiles with no
  defines, so the shader needs a default to compile standalone.
- **Types:** `true`/`false` becomes `bool`; an integer literal becomes `uint32_t`.
- **Integer literals follow C/Slang syntax**, and the generated field holds the same value Slang
  compiles: a leading `0` is octal (`010` → `8`) and `0x` is hexadecimal (`0x10` → `16`); the
  value is normalized to decimal in the generated code and in `defines()`. Suffixes such as
  `10u` or `10UL` are accepted.
- **A recognized declaration with a missing or non-literal default is a hard error.**
  `#define PICTOR_X` (no value), `#define PICTOR_X (1 << 3)` and `#define PICTOR_X OTHER` all
  fail generation with a diagnostic naming the macro. A `PICTOR_`-prefixed `#ifndef` that is not
  immediately followed by its matching `#define` is not recognized as a declaration and is
  skipped, so keep the three lines together.
- **The generated field name must be a valid C++ identifier.** `PICTOR_FOR` (`for`),
  `PICTOR_1` (`1`) and `PICTOR_` (empty) are rejected, as is a collision between two macros that
  map to the same field (for example `PICTOR_FOO` and `PICTOR_foo`); the diagnostic names the
  offending macro(s).
- **Comments and inactive preprocessor branches are ignored.** The scan strips comments before
  parsing and tracks `#if`/`#ifdef`/`#ifndef`/`#elif`/`#else`/`#endif`, so a commented-out or
  `#if 0` declaration never becomes a tunable. `#if` expressions are evaluated when they are an
  integer literal, `defined(NAME)`, a macro defined earlier in the same file, or a negation of
  one; any other expression is treated as active, so an exotic condition around a declaration may
  still be extracted. Keep conditions around declarations simple (`0`, `defined(...)`, or a
  declared macro).
- `pictor_reflect` scans the **whole dependency closure** (the entry file plus every
  `#include`/`import`), so a tunable declared in a shared module is found. Two files declaring the
  same macro must agree on the default, or generation fails.

### The generated `Tunables` struct

When a shader's closure declares at least one tunable, `pictor_reflect` emits one struct in the
`<stem>_shader` namespace (a shader with no tunables emits nothing):

```cpp
struct SortTunables {
    uint32_t vpt   = 16;   // PICTOR_VPT
    uint32_t waves = 8;    // PICTOR_WAVES

    [[nodiscard]] ShaderDefines defines() const;   // generated body
};
```

`ShaderDefines` is a small owning builder (`add(name, value)` / `view()`); `defines()` is
generated and emitted `inline`, so the header is safe to include from several translation units
and there is no hand-written mapping to keep in sync:

```cpp
inline ShaderDefines SortTunables::defines() const
{
    ShaderDefines d;
    d.add("PICTOR_VPT", std::to_string(vpt));
    d.add("PICTOR_WAVES", std::to_string(waves));
    return d;
}
```

- **Field-name mapping:** the macro name lower-cased with the `PICTOR_` prefix removed —
  `PICTOR_VPT` → `vpt`, `PICTOR_WAVE_MATCH` → `wave_match`. Names that do not map to a valid,
  collision-free C++ identifier are rejected at generation time (see the convention above).
- **Every trait aliases the struct.** Each entry trait gains `using Tunables = <Stem>Tunables;`
  so a tuner can name the type as `TCode::Tunables` (a shader with no tunables emits no alias,
  and the tuned `create` overload is not viable for it):

```cpp
struct SortCompute {
    static constexpr std::string_view entry_point = "sort_compute";
    static constexpr const pictor::ShaderSources& sources = sort_shader_sources::cSet;
    using Tunables = SortTunables;
};

using SortComputeExecutor = pictor::ComputeExecutor<SortCompute, SortParams>;
```
- **`ShaderDefines` is move-only.** A copy would leave the copied `string_view`s pointing into
  the source's storage, so the copy constructor and assignment are deleted; `defines()` returns
  by value and the generated code relies on the move. `view()` is a span over the builder's own
  storage: it is invalidated by any later `add()`, so build the defines first and view them last.
- **The defaults come from the shader**, so the host starts from what the shader declared and
  overrides only what it needs (integers are normalized to the value Slang compiles, for example
  octal `010` → `8`):

```cpp
sort_shader::SortTunables tunables;            // vpt = 16, waves = 8
tunables.vpt = pick_items_per_thread(...);
auto defines = tunables.defines();
auto exec = sort_shader::SortComputeExecutor::create(ctx, nullptr, {defines.view()});
```

### Injecting a tuner at executor construction

Choosing values is the runtime's job, so both executor factories take an optional tuner callable:

```cpp
// tuner: void(const pictor::DeviceInfo&, TCode::Tunables&)
auto exec = sort_shader::SortComputeExecutor::create(
    ctx, [](const pictor::DeviceInfo& device, sort_shader::SortTunables& tunables) {
        tunables.vpt = device.maxComputeSharedMemorySize >= 100 * 1024 ? 21u : 16u;
    });
```

That callable shape is a concept, `pictor::TunerFor<TCode, TTuner>`, and the shared step it feeds
is `pictor::tuned_defines<TCode>(device, tuner)` — both declared in
`include/pictor_core/execution/base_executor.hpp`, at the base of the executor hierarchy. The step
is common to every executor (it never depends on how a particular executor compiles or stages its
modules), so it lives there rather than being repeated per executor. The tuned `create` on each
executor is then only a constraint plus two lines.

- The executor default-constructs `TCode::Tunables`, so the shader's declared defaults are the
  starting point; the tuner overrides only what it knows.
- The tuner is called once, with `ctx.device_info` — the cached device facts and limits (identity,
  shared-memory and workgroup limits, storage alignment, subgroup size). See
  `include/pictor_core/context/device_info.hpp`.
- The executor then calls `defines()` and compiles with the resulting `ShaderDefines`, so a
  tunable that participates in `[numthreads]` is re-reflected: the returned executor's `wgX` (or
  `taskWgX` / `meshWgX`) is the value the tuner chose.
- `create(ctx, nullptr)` still selects the untuned overload. The tuned overload is constrained on
  its second argument being invocable as a tuner, so the two are never ambiguous, and only a
  shader that declares tunables can be tuned. A free function, a lambda, or a stateful tuner that
  caches per device all work.

**Device-limits contract.** The tuner receives the limits and **must** return a configuration
that fits the device. The executor does not model the kernel's shared-memory, register or
workgroup costs, so it cannot pre-validate the configuration: a config that exceeds a limit is at
best a failed `create` (`eErrorInitializationFailed` for a compile failure, otherwise the
`vk::Result` from pipeline creation) and at worst a driver crash — a 7680-item tile that needed
49256 B of workgroup shared memory against a 49152 B `maxComputeSharedMemorySize` produced
`VK_ERROR_DEVICE_LOST`. Cost models live next to the primitive, not in the executor (issue #13).

### A primitive's own tuner: the radix sort

`RadixSort` is the worked example of those rules. `algorithms/shaders/sort.slang` declares its
workgroup shape as tunables (`PICTOR_WAVES` / `PICTOR_VPT`, defaulting to the conservative `8`x`16`
that fits a 32 KiB shared-memory budget), and `algorithms/src/sort.cpp` chooses the shape per
device:

- a table of measured devices (`kMeasuredShapes`, keyed by vendor and device id) overrides the
  shader's declared shape — the RTX 4090 entry is `12`x`18`, the best measured shape on that
  hardware, while `8`x`16` measured ~17% faster than `12`x`18` on the RTX 2070 SUPER;
- the shape is scaled by the key width before it is fitted: the tile's staged bytes are what the
  shape controls, so a 64-bit key takes half the items per thread of the 32-bit baseline;
- `RadixSort::fit_to_device` then narrows the shape to the device's limits, and
  `RadixSort::shared_bytes` / `config_fits` expose the kernel's cost model so a caller picking its
  own shape can stay within `maxComputeSharedMemorySize` instead of finding out from the driver.
  The cost model is the sum of the bin kernel's `groupshared` arrays, updated next to the shader
  that declares them;
- the default `RadixSort::create(ctx)` runs that tuner, and `create(ctx, spec, SortShape{...})`
  takes an explicit shape (zero fields mean "pick for the device").

The shape is host state as well as shader state — the host sizes every dispatch and the scratch
layout from it — so the sort resolves the tunables once and compiles all four of its entry points
from the same defines, rather than tuning each executor separately. It also takes the block size
back from the reflected `wgX` instead of re-deriving `subgroup_size * waves`, so a `[numthreads]`
that ignored the tunable fails loudly rather than mis-sorting.

### Device-agnostic by design

The reflector extracts **names and defaults only**; it deliberately knows nothing about the
device. Choosing values, and validating them against `maxComputeSharedMemorySize`,
`maxComputeWorkGroupInvocations` and friends, is the runtime's job before the pipeline is
created — the tuned `create` overload injects that choice at executor construction (see
[Injecting a tuner at executor construction](#injecting-a-tuner-at-executor-construction)).
Anything that sizes an array or appears in `[numthreads]` must be a macro, so it forces a
recompile per configuration; for value tuning that does not change the module's capabilities,
`create` already accepts a `vk::SpecializationInfo*` (see
[Compiling SPIR-V yourself](#compiling-spir-v-yourself)).

## Device-side modules

Shaders may be split into modules and included files. Three forms work, and all of them are
resolved identically at build time and at run time:

```slang
import wave_ops;              // module by name, resolved through the search paths
import "ops/scale";           // module by path, relative to the importing file
#include "ops/include_helper.slang"  // textual include
```

- **Visibility.** Only `public` declarations are visible across a module boundary — mark
  functions and types you want to export `public`.
- **Search paths.** The shader's own directory is always first (`/sp0`); each
  `pictor_add_shader(... INCLUDE_DIRS <dir>...)` entry follows in order (`/sp1`, `/sp2`, ...).
  A relative import inside a module resolves against that module's own directory first, then the
  search paths.
- **Transitive closure.** Imports of imports and `#include`d files are collected into the closure
  and embedded, so a single `import` can pull in a whole library. The generated depfile lists
  every file, so editing any of them regenerates the headers.
- **Run time.** The whole closure is compiled from memory through an in-memory file system. No
  shader sources need to ship next to the binary, and the runtime performs no filesystem access;
  resolution matches build time because the same search-path mapping is embedded.
- **Module names.** A by-name import (`import util;`) resolves to the first search path that
  contains `util.slang`; imports by path can distinguish same-named files in different
  directories. Keep basenames unique across the closure to avoid silently resolving a by-name
  import to the wrong module.
- **No precompiled modules.** Modules are recompiled by the executor with the real `WAVE_SIZE`,
  so precompiled Slang modules are deliberately unsupported.

`examples/modules` shows all of it: `mix.slang` imports `wave_ops` through `INCLUDE_DIRS`,
`wave_ops.slang` imports `scale` relative to itself, and `scale.slang` holds a plain function.
`--dump` lists the closure:

```
module: examples/modules/mix.slang
entry mix_compute [compute] threadgroup 32x1x1
  push constant: MixParams
struct MixParams size 16 align 8
  out_addr @0 size 8 uint64_t
  count @8 size 4 uint32_t
sources: 3
  /sp0/mix.slang <- .../examples/modules/mix.slang
  /sp1/scale.slang <- .../examples/modules/ops/scale.slang
  /sp1/wave_ops.slang <- .../examples/modules/ops/wave_ops.slang
```

## Compiling SPIR-V yourself

If you already have SPIR-V — from `slangc`, a previous run, or an asset — bypass the source path:

```cpp
// Compute
auto exec = pictor::ComputeExecutor<MyCode, MyParams>::create_from_spirv(
    ctx, spirv, "main", nullptr, {64, 1, 1});

// Task + mesh
auto mesh = pictor::MeshExecutor<MyCode, MyParams>::create_from_spirv(
    ctx, taskSpirv, meshSpirv, "main", "main", nullptr, {32, 1, 1}, {64, 1, 1});
```

- `threadGroupSize` (compute) and `taskThreadGroupSize` / `meshThreadGroupSize` (mesh) are
  **inputs** here: `record()` divides the requested work-item counts by the workgroup size, and
  raw SPIR-V carries no reflection the executor can read. Pass the sizes the shader was compiled
  with; the defaults are `{1, 1, 1}`, which makes `record(x, y, z)` treat `x/y/z` as workgroup
  counts.
- The entry-name parameters exist for interface symmetry, but the pipeline stages are created with
  the name `"main"` — Slang emits entry points as `main` in SPIR-V. The passed names are currently
  ignored.
- `specInfo` is an optional `vk::SpecializationInfo*` applied to every stage.

## Host-side patterns

A complete dispatch loop, in order:

```cpp
auto ctx_res = Context::create();
if (!ctx_res.ok())
    return 1;
auto ctx = std::move(ctx_res.val());

auto exec_res = MyExecutor::create(*ctx);
if (!exec_res.ok())
    return 1;
auto exec = std::move(exec_res.val());

auto d_in  = ctx->pool_allocator->allocate(bytes, alignof(float));   // device, BDA
auto h_in  = ctx->pinned_allocator->allocate(bytes, alignof(float)); // host staging, mapped
std::memcpy(h_in.mappedPtr, data, bytes);
ctx->flush(h_in);
ctx->copy(h_in, d_in);
ctx->barrier(d_in, Access::TransferWrite, Access::Read);

MyParams params{};
params.in_addr  = d_in.address;
params.out_addr = d_out.address;
exec->record(count, 1, 1, params);

ctx->barrier(d_out, Access::Write, Access::TransferRead);
ctx->copy(d_out, h_out);
ctx->fsync();
ctx->invalidate(h_out);

ctx->deallocate(d_in);
ctx->deallocate(h_in);
// executor and Context destructors release the pipeline and the device
```

- **Errors.** Every factory returns `Expected<T, vk::Result>`: check `ok()`, use `val()`, and
  `err()` for the Vulkan result. No exceptions are used anywhere.
- **Barriers.** `Access::Read` / `Write` / `ReadWrite` cover compute, task and mesh stages;
  `TransferRead` / `TransferWrite` cover transfers; `IndirectRead` covers indirect commands. The
  library records exactly the barrier you ask for; it does not insert hidden ones.
- **Streams.** `ctx->create_stream()` returns a handle; pass it to `record`, `barrier`, `copy`,
  `fill` and `update`, then `ctx->submit(handle)` or `ctx->fsync(handle)`. `cDefaultStream` (0) is
  used when omitted. Streams are double-buffered command buffers with fences.
- **Host visibility.** `flush` after writing pinned memory, `invalidate` before reading it.
- **Teardown.** Destroy executors and deallocate buffers before the `Context` goes out of scope;
  the context's destructor waits idle and destroys the allocators.

## Reference

### `pictor_add_shader`

```cmake
pictor_add_shader(<target> <file.slang> [INCLUDE_DIRS <dir>...])
```

- Runs `pictor_reflect` on `<file.slang>` and generates `<stem>.shader.h` + `<stem>.slang.h` in
  `CMAKE_CURRENT_BINARY_DIR`.
- Creates an INTERFACE library `<target>` that carries the generated include directory and links
  `pictor_core`. Link it into the consuming target.
- Relative `INCLUDE_DIRS` entries are resolved against `CMAKE_CURRENT_SOURCE_DIR`; they add
  import search paths (in order).
- On Ninja and Makefile generators a depfile wires every imported/included file into the build
  graph, so editing a module regenerates dependent headers. Other generators only regenerate on
  reconfigure.
- Call it in the same `CMakeLists.txt` that defines the consuming target: the generated include
  path is attached to `<target>` and does not propagate to other directories.

### `pictor_reflect`

`pictor_add_shader` invokes the tool for you; it can also be run by hand (it is built into the
build tree as `build/pictor_reflect`):

```
pictor_reflect <shader.slang> --outdir <dir> [--wave-size <n>] [-I <dir>]... [--depfile <file>] [--dump]
```

| Option | Meaning |
|---|---|
| `<shader.slang>` | The entry shader. Its directory becomes search path `/sp0`. |
| `--outdir <dir>` | Output directory (created if needed). Defaults to the current directory. |
| `--wave-size <n>` | Subgroup size used for this reflection pass (default 32). It affects what the reflection sees (for example the threadgroup size shown by `--dump`) and the multi-layout push-constant resolution compile; the generated headers do not bake it in for runtime compilation, and the runtime recompiles with the real device value. A push-constant layout that depends on `WAVE_SIZE` is not supported. |
| `-I <dir>` | Additional import search path, repeatable. |
| `--depfile <file>` | Writes a Make-style dependency rule listing the whole closure. |
| `--dump` | Prints the reflection to stdout (see below) in addition to writing the headers. |

Exit status is 0 on success and 1 on any failure: bad usage or unknown option, module load or
link errors, no entry points found, a dependency outside the declared search paths, a guarded
tunable with a missing or non-literal default, conflicting guarded defaults for one macro, an
invalid or colliding generated field name, a push-constant resolution failure (multiple layouts
in one entry, a task/mesh layout mismatch, an imported block, or unreadable SPIR-V), or a write
failure. Diagnostics go to stderr; the emitted headers are only written on success.

Hand-run example:

```bash
./build/pictor_reflect examples/modules/mix.slang \
    --outdir scratch/out -I examples/modules/ops --dump
```

`--dump` output for `examples/compute/add.slang`:

```
module: examples/compute/add.slang
entry add_compute [compute] threadgroup 64x1x1
  push constant: AddParams
struct AddParams size 32 align 8
  a_addr @0 size 8 uint64_t
  b_addr @8 size 8 uint64_t
  out_addr @16 size 8 uint64_t
  count @24 size 4 uint32_t
sources: 1
  /sp0/add.slang <- .../examples/compute/add.slang
```

- `entry <name> [stage] threadgroup AxBxC` — every entry point, its stage, and the reflected
  workgroup size (compute, task and mesh only).
- `push constant: <Type>` — the push-constant struct bound to that entry.
- `binding set N index M <name> <type> x<count>` — descriptor bindings, when present.
- `struct <Name> size S align A` — each reachable struct with the exact field offsets, sizes and
  reflected types; `?` marks kinds the generator cannot map (they fall back to byte arrays).
- `fragment outputs: N` — color attachment count for fragment entries.
- `sources: N` — the embedded closure with virtual path and real path.
- `tunables: N` — each `PICTOR_`-prefixed guarded default found in the closure's live code
  (`NAME = default (type)`); omitted when the shader declares none.

## Limitations, FAQ, and troubleshooting

### Current limitations

- **Raster stages.** Vertex and fragment entries are reflected (`--dump`) but no traits or
  executors are emitted yet; the raster substrate is in progress.
- **Descriptor bindings.** Bindings are reflected (`--dump` shows them) but no resource struct is
  emitted yet; the push-descriptor generation comes with the raster work. Until then, pass buffers
  as BDA addresses in push constants.
- **Unsupported field kinds** become byte arrays (matrices, and anything else the generator cannot
  map), so they keep their size but lose their field names. Enums are not affected: they map to
  their underlying integer scalar.
- **Vector arrays** (`float3 v[N]`) are not reproduced in C++ (see
  [Layout caveats](#layout-caveats)); keep push-constant arrays scalar or `float2`/`float4`.
- **Bitfields** produce Slang's synthesized backing-field name.
- **No push constants** → `void` params → the executor alias does not compile when used. Declare
  at least one field in the params struct.
- **One push-constant layout per entry.** The executor carries one `TParams`; an entry that uses
  more than one distinct layout is rejected. Same-type blocks are one layout and are accepted. A
  task/mesh pair must resolve to the same layout (or one stage may use none).
- **Push constants must be declared in the entry file (or a `#include`d file).** A block in an
  `import`ed module is omitted from the entry's program layout; an entry that uses one is rejected
  unless the entry file declares the same layout, instead of being bound to `void`.
- **Descriptor bindings are still module-wide per entry.** Per-entry resource filtering belongs
  with the generated resource work; until then `--dump` lists every binding for every entry.
- **SPIR-V entry names** are ignored by `create_from_spirv`; the pipeline uses `"main"`.
- **Runtime diagnostics are dropped** by `Executor::create`, which reports
  `vk::Result::eErrorInitializationFailed`. To see the Slang messages, call
  `ctx.shader_compiler->compile(trait::sources, entry, stage, ctx.subgroup_size)` directly and
  read `CompileResult::diagnostics`.

### FAQ

**Do shader sources need to ship with the binary?** No. The closure is embedded in the generated
header and compiled from memory; the runtime never touches the filesystem.

**Can I use `slangc` instead?** Yes, but then you own the compile and pass the SPIR-V through
`create_from_spirv`, including the workgroup size.

**Why is my dispatch count wrong?** `record(x, y, z, ...)` treats `x/y/z` as work items and
divides by the reflected workgroup size. Use `record_raw` (mesh) or divide by `wgX`/`taskWgX`
yourself.

**How do I share code between shaders?** Put it in a module, `import` it, and add its directory
to `INCLUDE_DIRS`. Mark exported declarations `public`.

**Can I use `WAVE_SIZE` in `[numthreads]`?** Yes — it is defined when the executor compiles the
shader, from the device subgroup size.

### Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Build fails with `cannot open file 'x.slang'` | Import not found. Add the directory to `INCLUDE_DIRS` or fix the relative path. |
| Build fails with `dependency outside the search paths` | A resolved file is not under the shader's directory or an `INCLUDE_DIRS` entry. Move it under one of them. |
| Executor creation fails with `eErrorInitializationFailed` | Run-time compile or link failure; the executor hides the diagnostics. Compile the trait's `sources` through `ShaderCompiler` directly to read them. |
| `forming reference to void` when instantiating an executor | The shader has no push-constant struct (or an empty one). Declare at least one field. |
| Wrong values in later push-constant fields | An unsupported field shape (vector array, bitfield, pointer-to-vector) shifted the C++ layout. See [Layout caveats](#layout-caveats). |
| `--dump` shows `?` for a field type | The kind has no C++ mapping; the field is emitted as a byte array. |
| A module edit does not regenerate headers | On a non-Ninja/Makefile generator the depfile is not used; reconfigure, or switch generator. |
| Generation fails with `non-literal default` or `has no value` | A `PICTOR_`-prefixed guarded define must default to `true`/`false` or an integer literal; simplify the default. |
| Generation fails with `invalid C++ field name` | The macro name maps to a C++ keyword, an empty name or a name that is not an identifier; rename the macro. |
| Generation fails with `both map to the field` | Two macros lower-case to the same field name; rename one of them. |
| Generation fails with `conflicting guarded defaults` | Two files in the closure declare the same macro with different defaults; make them agree. |
| Generation fails with `distinct push-constant layouts` | An entry uses two different blocks; split the entry or pass one block. |
| Generation fails with `push constants must be declared in the entry file` | An entry uses a block declared in an `import`ed module whose layout is not declared in the entry file; move it into the entry file or a `#include`d file. |
| Two modules fight over the same name | A by-name import picks the first search path that has it; rename one of the files or import by path. |

## Performance

Measured on the development machine (RTX 2070 SUPER, GCC 14.3, Release):

| Stage | Cost |
|---|---|
| Build-time reflection (`pictor_reflect`) | ~0.12 s for a 2-file shader; ~0.20 s for a 21-file closure, per shader invocation. A module with no `push_constant` mention outside the entry file produces no SPIR-V. |
| Multi-layout push-constant disambiguation | one SPIR-V compile per entry; ~0.25 s extra for `bc.slang` (9 entries, 4 layouts). |
| Run-time executor creation (Slang JIT + pipeline) | ~22 ms for a single-file shader; ~45 ms for a 21-file closure, after the first compile in the process (the first one pays a Slang warm-up of roughly 2x). |

Implications:

- Compilation is paid once per executor at `create()`, never per dispatch. Create executors up
  front and reuse them.
- Build-time cost scales gently with the closure size; the module system adds no run-time
  filesystem or packaging cost. A module with multiple distinct push-constant layouts (or another
  closure file mentioning `push_constant`) pays one build-time compile per entry; modules with a
  single layout and no other `push_constant` mention are unchanged.
- The generated headers embed the sources verbatim, so they grow with the closure; the 21-file
  test closure is still only a few kilobytes of header.
