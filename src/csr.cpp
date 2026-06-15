/* AI GENERATED FILE  */

#include "csr.hpp"

#include "pictor_core/context/context.hpp"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <fcntl.h>
#include <string_view>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace pictor {
namespace graphs {

namespace {

struct MappedFile {
  void *data;
  size_t size;

  MappedFile(const std::string &filename) {
    int fd = open(filename.c_str(), O_RDONLY);
    if (fd == -1)
      throw std::runtime_error("Could not open file: " + filename);

    struct stat st;
    if (fstat(fd, &st) == -1) {
      close(fd);
      throw std::runtime_error("Could not stat file: " + filename);
    }

    size = (size_t)st.st_size;
    data = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);

    if (data == MAP_FAILED)
      throw std::runtime_error("Could not mmap file: " + filename);
  }

  ~MappedFile() {
    if (data != MAP_FAILED)
      munmap(data, size);
  }
};

const char *skip_comments(const char *ptr, const char *end) {
  while (ptr < end && *ptr == '%') {
    while (ptr < end && *ptr != '\n')
      ptr++;
    if (ptr < end)
      ptr++;
  }
  return ptr;
}

const char *next_line(const char *ptr, const char *end) {
  while (ptr < end && *ptr != '\n')
    ptr++;
  if (ptr < end)
    ptr++;
  return ptr;
}

struct MtxHeader {
  uint64_t rows, cols, edges;
  bool symmetric;
  const char *data_start;
};

MtxHeader parse_header(const char *start, const char *end) {
  const char *first_line_end = next_line(start, end);
  std::string_view header_line(start, (size_t)(first_line_end - start));
  bool symmetric = header_line.find("symmetric") != std::string_view::npos;

  const char *ptr = skip_comments(start, end);
  MtxHeader header;
  header.symmetric = symmetric;

  auto res = std::from_chars(ptr, end, header.rows);
  ptr = res.ptr;
  while (ptr < end && isspace(*ptr))
    ptr++;
  res = std::from_chars(ptr, end, header.cols);
  ptr = res.ptr;
  while (ptr < end && isspace(*ptr))
    ptr++;
  res = std::from_chars(ptr, end, header.edges);
  ptr = res.ptr;
  header.data_start = next_line(ptr, end);

  return header;
}

} // namespace

Graph::Graph(Context *ctx) : ctx(ctx) {}
Graph::~Graph() {
  if (ctx) {
    ctx->deallocate(h_row_offsets);
    ctx->deallocate(h_column_indices);
    ctx->deallocate(d_row_offsets);
    ctx->deallocate(d_column_indices);
  }
}

void Graph::save_to_binary(const std::string &prefix) const {
  throw std::string{"not implemented"};
}

void Graph::load_from_binary(const std::string &prefix) {
  throw std::string{"not implemented"};
}

Graph Graph::import(Context &ctx, std::string const &file, bool symmetrize) {
  MappedFile mf(file);
  const char *start = static_cast<const char *>(mf.data);
  const char *end = start + mf.size;

  auto header = parse_header(start, end);
  bool force_symmetric = symmetrize || header.symmetric;

  uint64_t num_vertices = std::max(header.rows, header.cols);

  Graph g(&ctx);
  g.num_vertices = num_vertices;

  std::vector<std::atomic<uint64_t>> degrees(num_vertices + 1);
  for (auto &d : degrees)
    d.store(0);

  unsigned int num_threads = std::thread::hardware_concurrency();
  std::vector<std::thread> threads;

  auto process_chunk_count = [&](const char *c_start, const char *c_end) {
    const char *ptr = c_start;
    while (ptr < c_end) {
      uint64_t u, v;
      auto res = std::from_chars(ptr, c_end, u);
      if (res.ec != std::errc{})
        break;
      ptr = res.ptr;
      while (ptr < c_end && isspace(*ptr))
        ptr++;
      res = std::from_chars(ptr, c_end, v);
      if (res.ec != std::errc{})
        break;
      ptr = next_line(res.ptr, c_end);

      if (u > 0 && u <= num_vertices)
        degrees[u - 1].fetch_add(1, std::memory_order_relaxed);
      if (force_symmetric && u != v) {
        if (v > 0 && v <= num_vertices)
          degrees[v - 1].fetch_add(1, std::memory_order_relaxed);
      }
    }
  };

  const char *data_ptr = header.data_start;
  size_t data_size = (size_t)(end - data_ptr);
  for (unsigned int i = 0; i < num_threads; ++i) {
    const char *c_start = data_ptr + (i * data_size / num_threads);
    const char *c_end = data_ptr + ((i + 1) * data_size / num_threads);

    if (i > 0)
      c_start = next_line(c_start, end);
    if (i < num_threads - 1)
      c_end = next_line(c_end, end);
    else
      c_end = end;

    if (c_start < c_end)
      threads.emplace_back(process_chunk_count, c_start, c_end);
  }

  for (auto &t : threads)
    t.join();
  threads.clear();

  // Allocate host pinned memory for row offsets
  g.h_row_offsets = ctx.pinned_allocator->allocate(
      (num_vertices + 1) * sizeof(uint64_t), alignof(uint64_t));
  g.row_offsets = std::span<uint64_t>(
      static_cast<uint64_t *>(g.h_row_offsets.mappedPtr), num_vertices + 1);

  uint64_t current_offset = 0;
  for (uint64_t i = 0; i < num_vertices; ++i) {
    g.row_offsets[i] = current_offset;
    current_offset += degrees[i].load(std::memory_order_relaxed);
  }
  g.row_offsets[num_vertices] = current_offset;
  g.num_edges = current_offset;

  // Allocate host pinned memory for column indices
  g.h_column_indices = ctx.pinned_allocator->allocate(
      g.num_edges * sizeof(uint32_t), alignof(uint32_t));
  g.column_indices = std::span<uint32_t>(
      static_cast<uint32_t *>(g.h_column_indices.mappedPtr), g.num_edges);

  // Use atomic counters for filling column indices in parallel
  // Reuse degrees vector to store current fill positions (relative to
  // row_offsets[i])
  for (auto &d : degrees)
    d.store(0);

  auto process_chunk_fill = [&](const char *c_start, const char *c_end) {
    const char *ptr = c_start;
    while (ptr < c_end) {
      uint64_t u, v;
      auto res = std::from_chars(ptr, c_end, u);
      if (res.ec != std::errc{})
        break;
      ptr = res.ptr;
      while (ptr < c_end && isspace(*ptr))
        ptr++;
      res = std::from_chars(ptr, c_end, v);
      if (res.ec != std::errc{})
        break;
      ptr = next_line(res.ptr, c_end);

      if (u > 0 && u <= num_vertices) {
        uint64_t pos = g.row_offsets[u - 1] +
                       degrees[u - 1].fetch_add(1, std::memory_order_relaxed);
        g.column_indices[pos] = (uint32_t)(v - 1);
      }
      if (force_symmetric && u != v) {
        if (v > 0 && v <= num_vertices) {
          uint64_t pos = g.row_offsets[v - 1] +
                         degrees[v - 1].fetch_add(1, std::memory_order_relaxed);
          g.column_indices[pos] = (uint32_t)(u - 1);
        }
      }
    }
  };

  for (unsigned int i = 0; i < num_threads; ++i) {
    const char *c_start = data_ptr + (i * data_size / num_threads);
    const char *c_end = data_ptr + ((i + 1) * data_size / num_threads);

    if (i > 0)
      c_start = next_line(c_start, end);
    if (i < num_threads - 1)
      c_end = next_line(c_end, end);
    else
      c_end = end;

    if (c_start < c_end)
      threads.emplace_back(process_chunk_fill, c_start, c_end);
  }

  for (auto &t : threads)
    t.join();

  // Upload to device
  g.d_row_offsets =
      ctx.pool_allocator->allocate(g.h_row_offsets.size, alignof(uint64_t));
  g.d_column_indices =
      ctx.pool_allocator->allocate(g.h_column_indices.size, alignof(uint32_t));

  StreamHandle stream = ctx.create_stream();
  ctx.copy(g.h_row_offsets, g.d_row_offsets, stream);
  ctx.copy(g.h_column_indices, g.d_column_indices, stream);
  ctx.flush(stream);
  ctx.sync(stream);

  return g;
}

} // namespace graphs
} // namespace pictor
