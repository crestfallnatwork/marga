#include "bfs.hpp"
#include "pictor_core/context/context.hpp"
#include "pictor_core/execution/shader_compiler.hpp"
#include "pictor_core/execution/synchronization.hpp"

// Generated headers from Slang
#include "bfs.layout.hpp"
#include "bfs.slang.h"

#include <chrono>
#include <fstream>
#include <iostream>

namespace pictor {
namespace graphs {

namespace {
std::string load_shader_file(const std::string &path) {
  std::ifstream file(path);
  if (!file.is_open())
    return "";
  return std::string((std::istreambuf_iterator<char>(file)),
                     std::istreambuf_iterator<char>());
}
} // namespace

BFS::BFS(Context &ctx) : _ctx(ctx) {
  auto compiler_res = ShaderCompiler::create();
  if (!compiler_res.ok())
    throw std::runtime_error("Failed to create shader compiler");
  auto &compiler = *compiler_res.val();

  std::string source(reinterpret_cast<char *>(bfs_slang), bfs_slang_len);

  auto task_res = compiler.compile(source, "task", ShaderStage::Task);
  auto mesh_res = compiler.compile(source, "mesh", ShaderStage::Mesh);
  auto indirect_res =
      compiler.compile(source, "compute_indirect", ShaderStage::Compute);

  if (!task_res.success || !mesh_res.success || !indirect_res.success) {
    std::cerr << "Shader compilation failed!" << std::endl;
    std::cerr << "Task: " << task_res.diagnostics << std::endl;
    std::cerr << "Mesh: " << mesh_res.diagnostics << std::endl;
    std::cerr << "Indirect: " << indirect_res.diagnostics << std::endl;
    throw std::runtime_error("Shader compilation failed");
  }

  auto mesh_exec_res = MeshExecutor<bfs_json_layout::Params>::create_from_spirv(
      _ctx, task_res.spirv, mesh_res.spirv, "main", "main");
  if (!mesh_exec_res.ok())
    throw std::runtime_error("Failed to create mesh executor");
  _mesh_exec = std::move(mesh_exec_res.val());

  auto indirect_exec_res =
      ComputeExecutor<bfs_json_layout::Params>::create_from_spirv(
          _ctx, indirect_res.spirv, "main");
  if (!indirect_exec_res.ok())
    throw std::runtime_error("Failed to create indirect executor");
  _indirect_exec = std::move(indirect_exec_res.val());
}

BFS::~BFS() { _cleanup_buffers(); }

void BFS::_allocate_buffers(uint64_t num_vertices, uint64_t num_edges) {
  _cleanup_buffers();

  _frontier_a = _ctx.pool_allocator->allocate(num_vertices * sizeof(uint32_t),
                                              alignof(uint32_t));
  _frontier_b = _ctx.pool_allocator->allocate(num_vertices * sizeof(uint32_t),
                                              alignof(uint32_t));
  _frontier_size_a =
      _ctx.pool_allocator->allocate(sizeof(uint32_t), alignof(uint32_t));
  _frontier_size_b =
      _ctx.pool_allocator->allocate(sizeof(uint32_t), alignof(uint32_t));

  uint64_t visited_size = (num_vertices + 31) / 32 * sizeof(uint32_t);
  _visited = _ctx.pool_allocator->allocate(visited_size, alignof(uint32_t));
  _distances = _ctx.pool_allocator->allocate(num_vertices * sizeof(uint32_t),
                                             alignof(uint32_t));
  _h_distances = _ctx.pinned_allocator->allocate(
      num_vertices * sizeof(uint32_t), alignof(uint32_t));
  _indirect_buffer =
      _ctx.pool_allocator->allocate(4 * sizeof(uint32_t), alignof(uint32_t));
  _staging_size =
      _ctx.pinned_allocator->allocate(sizeof(uint32_t), alignof(uint32_t));
}

void BFS::_cleanup_buffers() {
  if (_frontier_a.allocation)
    _ctx.deallocate(_frontier_a);
  if (_frontier_b.allocation)
    _ctx.deallocate(_frontier_b);
  if (_frontier_size_a.allocation)
    _ctx.deallocate(_frontier_size_a);
  if (_frontier_size_b.allocation)
    _ctx.deallocate(_frontier_size_b);
  if (_visited.allocation)
    _ctx.deallocate(_visited);
  if (_distances.allocation)
    _ctx.deallocate(_distances);
  if (_h_distances.allocation)
    _ctx.deallocate(_h_distances);
  if (_indirect_buffer.allocation)
    _ctx.deallocate(_indirect_buffer);
  if (_staging_size.allocation)
    _ctx.deallocate(_staging_size);
}

std::pair<std::span<uint32_t>, BFSTimings> BFS::run(Graph &g,
                                                    uint32_t src_vertex) {
  auto t_full_start = std::chrono::high_resolution_clock::now();
  BFSTimings t;

  auto t_alloc_start = std::chrono::high_resolution_clock::now();
  if (_h_distances.size != g.num_vertices * sizeof(uint32_t)) {
    _allocate_buffers(g.num_vertices, g.num_edges);
  }
  auto t_alloc_end = std::chrono::high_resolution_clock::now();
  t.alloc_ms =
      std::chrono::duration<double, std::milli>(t_alloc_end - t_alloc_start)
          .count();

  StreamHandle stream = cDefaultStream;

  auto t_init_start = std::chrono::high_resolution_clock::now();
  _ctx.fill(_distances, 0, _distances.size, 0xFFFFFFFF, stream);
  _ctx.fill(_visited, 0, _visited.size, 0, stream);
  _ctx.fill(_frontier_size_a, 0, _frontier_size_a.size, 0, stream);
  _ctx.fill(_frontier_size_b, 0, _frontier_size_b.size, 0, stream);

  uint32_t one = 1;
  uint32_t zero = 0;
  _ctx.update(_frontier_a, 0, sizeof(uint32_t), &src_vertex, stream);
  _ctx.update(_frontier_size_a, 0, sizeof(uint32_t), &one, stream);
  _ctx.update(_distances, src_vertex * sizeof(uint32_t), sizeof(uint32_t),
              &zero, stream);

  uint32_t init_cmd[3] = {1, 1, 1};
  _ctx.update(_indirect_buffer, 0, sizeof(init_cmd), init_cmd, stream);

  uint32_t word_idx = src_vertex / 32;
  uint32_t mask = 1 << (src_vertex % 32);
  _ctx.update(_visited, word_idx * sizeof(uint32_t), sizeof(uint32_t), &mask,
              stream);

  _ctx.flush(stream);
  _ctx.sync(stream);
  auto t_init_end = std::chrono::high_resolution_clock::now();
  t.init_ms =
      std::chrono::duration<double, std::milli>(t_init_end - t_init_start)
          .count();

  uint32_t current_level = 0;
  bool finished = false;
  uint32_t batch = 8;
  uint32_t next_size = 1;

  auto t_traversal_start = std::chrono::high_resolution_clock::now();

  while (next_size > 0) {
    AllocationInfo *last_next_f_size = nullptr;
    for (uint32_t i = 0; i < batch; ++i) {
      bool use_a = (current_level % 2 == 0);
      AllocationInfo &curr_f = use_a ? _frontier_a : _frontier_b;
      AllocationInfo &next_f = use_a ? _frontier_b : _frontier_a;
      AllocationInfo &curr_f_size = use_a ? _frontier_size_a : _frontier_size_b;
      AllocationInfo &next_f_size = use_a ? _frontier_size_b : _frontier_size_a;

      _ctx.fill(next_f_size, 0);
      _ctx.barrier(next_f_size, Access::TransferWrite, Access::Read);

      auto params =
          bfs_json_layout::Params{.current_level = current_level,
                                  .row_offsets = g.d_row_offsets.address,
                                  .column_indices = g.d_column_indices.address,
                                  .visited = _visited.address,
                                  .distances = _distances.address,
                                  .current_frontier = curr_f.address,
                                  .next_frontier = next_f.address,
                                  .current_frontier_size = curr_f_size.address,
                                  .next_frontier_size = next_f_size.address,
                                  .indirect_buffer = _indirect_buffer.address,
                                  .num_vertices = g.num_vertices};

      _mesh_exec->record_indirect(_indirect_buffer, params);

      _ctx.barrier(
          {{.resource = &_visited, .from = Access::Write, .to = Access::Read},
           {.resource = &_distances, .from = Access::Write, .to = Access::Read},
           {.resource = &curr_f, .from = Access::Write, .to = Access::Read},
           {.resource = &next_f, .from = Access::Write, .to = Access::Read},
           {.resource = &next_f_size,
            .from = Access::Write,
            .to = Access::Read}});

      _indirect_exec->record(1, 1, 1, params);
      _ctx.barrier(_indirect_buffer, Access::Write, Access::IndirectRead);

      last_next_f_size = &next_f_size;
      current_level++;

      _ctx.copy(*last_next_f_size, _staging_size);
    }

    _ctx.flush(stream);

    auto t_sync_start = std::chrono::high_resolution_clock::now();
    _ctx.sync(stream);
    auto t_sync_end = std::chrono::high_resolution_clock::now();
    t.inner_sync_ms +=
        std::chrono::duration<double, std::milli>(t_sync_end - t_sync_start)
            .count();

    next_size = *(uint32_t *)_staging_size.mappedPtr;
  }

  auto t_traversal_end = std::chrono::high_resolution_clock::now();
  t.traversal_ms = std::chrono::duration<double, std::milli>(t_traversal_end -
                                                             t_traversal_start)
                       .count();

  auto t_final_readback_start = std::chrono::high_resolution_clock::now();
  _ctx.copy(_distances, _h_distances, stream);
  _ctx.flush(stream);
  _ctx.sync(stream);
  auto t_final_readback_end = std::chrono::high_resolution_clock::now();
  t.pci_readback_ms = std::chrono::duration<double, std::milli>(
                          t_final_readback_end - t_final_readback_start)
                          .count();

  auto t_full_end = std::chrono::high_resolution_clock::now();
  t.total_ms =
      std::chrono::duration<double, std::milli>(t_full_end - t_full_start)
          .count();

  return {std::span<uint32_t>(static_cast<uint32_t *>(_h_distances.mappedPtr),
                              g.num_vertices),
          t};
}

} // namespace graphs
} // namespace pictor
