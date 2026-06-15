/*
 * Adapted from GAP Benchmark Suite (Scott Beamer)
 * Adapted by Gemini CLI
 */

#include "bfs_validation.hpp"
#include <atomic>
#include <chrono>
#include <iostream>
#include <vector>
#include <algorithm>
#include <omp.h>

namespace pictor {
namespace graphs {

namespace {

// Simple Bitmap implementation matching GAPS
struct Bitmap {
  size_t size;
  uint64_t* data;

  Bitmap(size_t n) : size(n) {
    size_t num_words = (n + 63) / 64;
    data = new uint64_t[num_words];
    reset();
  }

  ~Bitmap() {
    delete[] data;
  }

  void set_bit_atomic(size_t i) {
    uint64_t mask = 1ULL << (i % 64);
    __sync_fetch_and_or(&data[i / 64], mask);
  }

  void set_bit(size_t i) {
    uint64_t mask = 1ULL << (i % 64);
    data[i / 64] |= mask;
  }

  bool get_bit(size_t i) const {
    uint64_t mask = 1ULL << (i % 64);
    return (data[i / 64] & mask) != 0;
  }

  void reset() {
    size_t num_words = (size + 63) / 64;
    #pragma omp parallel for
    for (size_t i = 0; i < num_words; i++) data[i] = 0;
  }

  void swap(Bitmap& other) {
    std::swap(data, other.data);
    std::swap(size, other.size);
  }
};

// Simplified SlidingQueue for porting GAPS logic
struct SlidingQueue {
    std::vector<uint32_t> data;
    size_t head = 0;
    size_t tail = 0;
    size_t next_tail = 0;

    SlidingQueue(size_t n) : data(n) {}

    void push_back(uint32_t v) {
        size_t pos = __sync_fetch_and_add(&next_tail, 1);
        data[pos] = v;
    }

    bool empty() const {
        return head == tail;
    }

    size_t size() const {
        return tail - head;
    }

    void slide_window() {
        head = tail;
        tail = next_tail;
    }

    uint32_t* begin() { return &data[head]; }
    uint32_t* end() { return &data[tail]; }
    const uint32_t* begin() const { return &data[head]; }
    const uint32_t* end() const { return &data[tail]; }
};

int64_t BUStep(const Graph &g, std::vector<int32_t> &parent, Bitmap &front,
               Bitmap &next, uint32_t current_level, std::vector<uint32_t>& distances) {
  int64_t awake_count = 0;
  next.reset();
  #pragma omp parallel for reduction(+ : awake_count) schedule(dynamic, 1024)
  for (uint32_t u=0; u < g.num_vertices; u++) {
    if (parent[u] < 0) {
      for (uint64_t i = g.row_offsets[u]; i < g.row_offsets[u+1]; i++) {
        uint32_t v = g.column_indices[i];
        if (front.get_bit(v)) {
          parent[u] = v;
          distances[u] = current_level + 1;
          awake_count++;
          next.set_bit(u);
          break;
        }
      }
    }
  }
  return awake_count;
}

int64_t TDStep(const Graph &g, std::vector<int32_t> &parent,
               SlidingQueue &queue, uint32_t current_level, std::vector<uint32_t>& distances) {
  int64_t scout_count = 0;
  #pragma omp parallel
  {
    #pragma omp for reduction(+ : scout_count) nowait
    for (auto q_iter = queue.begin(); q_iter < queue.end(); q_iter++) {
      uint32_t u = *q_iter;
      for (uint64_t i = g.row_offsets[u]; i < g.row_offsets[u+1]; i++) {
        uint32_t v = g.column_indices[i];
        int32_t curr_val = parent[v];
        if (curr_val < 0) {
          if (__sync_bool_compare_and_swap(&parent[v], curr_val, (int32_t)u)) {
            distances[v] = current_level + 1;
            queue.push_back(v);
            scout_count += -curr_val;
          }
        }
      }
    }
  }
  return scout_count;
}

void QueueToBitmap(const SlidingQueue &queue, Bitmap &bm) {
  #pragma omp parallel for
  for (auto q_iter = queue.begin(); q_iter < queue.end(); q_iter++) {
    bm.set_bit_atomic(*q_iter);
  }
}

void BitmapToQueue(const Graph &g, const Bitmap &bm,
                   SlidingQueue &queue) {
  #pragma omp parallel for
  for (uint32_t n=0; n < g.num_vertices; n++) {
    if (bm.get_bit(n))
      queue.push_back(n);
  }
  queue.slide_window();
}

} // namespace

std::pair<std::vector<uint32_t>, ValidationTimings> BFSValidator::run(const Graph& graph, uint32_t source) {
    ValidationTimings timings;
    auto start = std::chrono::high_resolution_clock::now();

    uint64_t n = graph.num_vertices;
    std::vector<uint32_t> distances(n, 0xFFFFFFFF);
    std::vector<int32_t> parent(n);
    
    #pragma omp parallel for
    for (uint32_t i = 0; i < n; i++) {
        uint32_t deg = graph.row_offsets[i+1] - graph.row_offsets[i];
        parent[i] = deg != 0 ? -(int32_t)deg : -1;
    }

    distances[source] = 0;
    parent[source] = source;

    SlidingQueue queue(n);
    queue.push_back(source);
    queue.slide_window();

    Bitmap curr(n);
    Bitmap front(n);

    int64_t edges_to_check = graph.num_edges;
    int64_t scout_count = graph.row_offsets[source+1] - graph.row_offsets[source];
    
    uint32_t current_level = 0;
    int alpha = 15, beta = 18;

    while (!queue.empty()) {
        if (scout_count > edges_to_check / alpha) {
            int64_t awake_count, old_awake_count;
            QueueToBitmap(queue, front);
            awake_count = queue.size();
            queue.slide_window();
            do {
                old_awake_count = awake_count;
                awake_count = BUStep(graph, parent, front, curr, current_level, distances);
                front.swap(curr);
                current_level++;
            } while ((awake_count >= old_awake_count) ||
                     (awake_count > (int64_t)n / beta));
            
            // Note: BitmapToQueue resets the queue and fills it from the bitmap
            // GAPS uses a specific method, here I'll just clear and refill.
            queue.head = 0; queue.tail = 0; queue.next_tail = 0;
            BitmapToQueue(graph, front, queue);
            scout_count = 1;
        } else {
            edges_to_check -= scout_count;
            scout_count = TDStep(graph, parent, queue, current_level, distances);
            queue.slide_window();
            current_level++;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    timings.traversal_ms = std::chrono::duration<double, std::milli>(end - start).count();

    return {distances, timings};
}

bool BFSValidator::verify(const Graph& graph, uint32_t source, 
                         std::span<const uint32_t> gpu_distances, 
                         std::span<const uint32_t> cpu_distances) {
    uint32_t mismatches = 0;
    uint32_t gpu_reachable = 0;
    uint32_t cpu_reachable = 0;

    for (uint32_t i = 0; i < graph.num_vertices; ++i) {
        if (gpu_distances[i] != 0xFFFFFFFF) gpu_reachable++;
        if (cpu_distances[i] != 0xFFFFFFFF) cpu_reachable++;

        if (gpu_distances[i] != cpu_distances[i]) {
            if (mismatches < 10) {
                std::cerr << "Mismatch at vertex " << i << ": GPU=" << gpu_distances[i] 
                          << ", CPU=" << cpu_distances[i] << std::endl;
            }
            mismatches++;
        }
    }

    std::cout << "Verification Results:" << std::endl;
    std::cout << "  CPU Reachable: " << cpu_reachable << std::endl;
    std::cout << "  GPU Reachable: " << gpu_reachable << std::endl;
    
    if (mismatches == 0) {
        std::cout << "SUCCESS: GPU distances match CPU baseline." << std::endl;
        return true;
    } else {
        std::cout << "FAILURE: " << mismatches << " distance mismatches found." << std::endl;
        return false;
    }
}

} // namespace graphs
} // namespace pictor
