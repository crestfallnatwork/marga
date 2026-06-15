# PageRank Optimization Architecture Plan

This document outlines the three-phase approach to elevate our Dense PageRank implementation to a SOTA, delta-stepped, fully batched pipeline.

## Phase 1: Intermediate Reduction (Atomic-Free Scatter/Gather)
**Goal:** Eliminate the dependency on `shaderInt64Atomics` (which is not universally supported) and solve the floating-point determinism issue structurally rather than relying on 64-bit integer atomics.

**Architecture Changes:**
1. **The `intermediate` Buffer:** We allocate a buffer of `PartialSum` structs sized exactly to the maximum number of Mesh Workgroups we can dispatch.
2. **Mesh Shader (Scatter):** 
   - Instead of `InterlockedAdd(pr_next[v], pulled_pr)`, the Mesh Shader writes its locally reduced wave sum to `intermediate[gl_WorkGroupID.x]`.
   - The output includes both the `destination_vertex` and the `pulled_pr` value.
   - *Result: Zero atomics, perfectly coalesced VRAM writes.*
3. **Compute Shader (Reduce):**
   - We introduce a new `reduce` entry point.
   - A Compute grid sweeps over the `intermediate` buffer. It uses warp-level segmented scans (`WavePrefixSum`) to aggregate adjacent partial sums that belong to the same destination vertex.
   - It writes the final reduced sum to a temporary buffer or directly integrates it with the Finalize stage.

## Phase 2: Delta-Stepping (Active Vertex Tracking)
**Goal:** Implement convergence checking so the algorithm scales down work exponentially in later iterations, processing only vertices whose PageRank is actively changing.

**Architecture Changes:**
1. **Convergence Parameter:** Add `float tolerance` to the CLI and Push Constants (e.g., `1e-6`).
2. **Finalize & Check Shader:**
   - After calculating `new_pr`, compute `delta = abs(new_pr - old_pr)`.
   - If `delta > tolerance`, the vertex is "active".
   - Atomically increment a global `active_count_next` and append the vertex ID to the `active_vertices_next` array.
3. **Data Structure:** We need two active vertex arrays (`active_a`, `active_b`) to ping-pong between iterations.

## Phase 3: Batched Indirect Dispatch (Zero CPU Synchronization)
**Goal:** Remove the `_ctx.sync(stream)` from the CPU loop, allowing the GPU to run the entire 20+ iteration pipeline without stalling for the host.

**Architecture Changes:**
1. **Prepare Indirect Shader:**
   - A tiny 1-thread compute shader runs at the end of each iteration.
   - It reads `active_count_next`.
   - It writes `DispatchIndirectCommand` for the Task Shader: `x = (active_count_next + 31) / 32`.
2. **C++ Command Recording:**
   - In `pagerank.cpp`, we unroll the loop during command recording.
   - For `iter = 0 to max_iterations`:
     - Dispatch Dangling.
     - Indirect Dispatch Task/Mesh.
     - Dispatch Reduce.
     - Dispatch Finalize/Check.
     - Dispatch Prepare Indirect.
     - Insert `vkCmdPipelineBarrier` between each step.
   - We execute this massive command buffer *once* and call `_ctx.sync()` *once* at the very end.
3. **Early Exit (Optional):** We can copy `active_count_next` to a host-visible mapped buffer and check it from the CPU periodically, or let the GPU run empty dispatches (which cost <1us) if `active_count` hits 0 before `max_iterations`.

---
*End of Plan*
