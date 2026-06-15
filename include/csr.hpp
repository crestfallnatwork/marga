#pragma once

#include "pictor_core/memory/allocation_type.hpp"

#include <cstdint>
#include <span>
#include <string>

namespace pictor {
struct Context;

namespace graphs {
struct Graph {
  ~Graph();
  Graph(Context *ctx);

  AllocationInfo d_row_offsets;
  AllocationInfo d_column_indices;

  uint64_t num_vertices = 0;
  uint64_t num_edges = 0;

  std::span<uint64_t> row_offsets;
  std::span<uint32_t> column_indices;

  void save_to_binary(const std::string &prefix) const;
  void load_from_binary(const std::string &prefix);

  static Graph import(Context &ctx, std::string const &file,
                      bool symmetrize = false);

private:
  AllocationInfo h_row_offsets;
  AllocationInfo h_column_indices;
  Context *ctx;
};
} // namespace graphs
} // namespace pictor
