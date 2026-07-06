/* AI GENERATED FILE  */
/* SPDX-License-Identifier: Unlicense */

/*
 * This is free and unencumbered software released into the public domain.
 *
 * Anyone is free to copy, modify, publish, use, compile, sell, or
 * distribute this software, either in source code form or as a compiled
 * binary, for any purpose, commercial or non-commercial, and by any
 * means.
 *
 * In jurisdictions that recognize copyright laws, the author or authors
 * of this software dedicate any and all copyright interest in the
 * software to the public domain. We make this dedication for the benefit
 * of the public at large and to the detriment of our heirs and
 * successors. We intend this dedication to be an overt act of
 * relinquishment in perpetuity of all present and future rights to this
 * software under copyright law.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 *
 * For more information, please refer to <http://unlicense.org/>
 */

#include "csr.hpp"
#include "helper.hpp"

#include "pictor_core/context/context.hpp"

#include <algorithm>
#include <charconv>
#include <fcntl.h>
#include <iostream>
#include <string_view>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <chrono>
#include <unistd.h>
#include <vector>

namespace pictor {
namespace graphs {

namespace {

struct MappedFile {
    void* data;
    size_t size;

    MappedFile(const std::string& filename)
    {
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

    ~MappedFile()
    {
        if (data != MAP_FAILED)
            munmap(data, size);
    }
};

const char* skip_comments(const char* ptr, const char* end)
{
    while (ptr < end && *ptr == '%') {
        while (ptr < end && *ptr != '\n')
            ptr++;
        if (ptr < end)
            ptr++;
    }
    return ptr;
}

const char* next_line(const char* ptr, const char* end)
{
    while (ptr < end && *ptr != '\n')
        ptr++;
    if (ptr < end)
        ptr++;
    return ptr;
}

struct MtxHeader {
    uint64_t rows, cols, edges;
    bool symmetric;
    const char* data_start;
};

MtxHeader parse_header(const char* start, const char* end)
{
    const char* first_line_end = next_line(start, end);
    std::string_view header_line(start, (size_t)(first_line_end - start));
    bool symmetric = header_line.find("symmetric") != std::string_view::npos;

    const char* ptr = skip_comments(start, end);
    MtxHeader header;
    header.symmetric = symmetric;

    auto res = std::from_chars(ptr, end, header.rows);
    ptr      = res.ptr;
    while (ptr < end && isspace(*ptr))
        ptr++;
    res = std::from_chars(ptr, end, header.cols);
    ptr = res.ptr;
    while (ptr < end && isspace(*ptr))
        ptr++;
    res               = std::from_chars(ptr, end, header.edges);
    ptr               = res.ptr;
    header.data_start = next_line(ptr, end);

    return header;
}

} // namespace

Graph::Graph(Context* ctx) : ctx(ctx) {}
Graph::~Graph()
{
    if (ctx) {
        ctx->deallocate(h_row_offsets);
        ctx->deallocate(h_column_indices);
        ctx->deallocate(d_row_offsets);
        ctx->deallocate(d_column_indices);
    }
}

void Graph::save_to_binary(const std::string& prefix) const
{
    throw std::string{"not implemented"};
}

void Graph::load_from_binary(const std::string& prefix)
{
    throw std::string{"not implemented"};
}


Graph Graph::import(Context& ctx, std::string const& file, bool symmetrize,
                    GraphLayout target_layout)
{
    MappedFile mf(file);
    const char* start = static_cast<const char*>(mf.data);
    const char* end   = start + mf.size;

    auto header          = parse_header(start, end);
    bool force_symmetric = symmetrize || header.symmetric;

    uint64_t num_vertices = std::max(header.rows, header.cols);

    Graph g(&ctx);
    g.num_vertices = num_vertices;
    g.layout       = target_layout;

    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 1;
    
    // Allocate local degrees for each thread to avoid atomic contention
    std::vector<std::vector<uint64_t>> local_degrees(
        num_threads, std::vector<uint64_t>(num_vertices + 1, 0));

    std::vector<std::thread> threads;

    auto process_chunk_count = [&](const char* c_start, const char* c_end, unsigned int tid) {
        const char* ptr = c_start;
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

            if (target_layout == GraphLayout::CSC) {
                if (v > 0 && v <= num_vertices)
                    local_degrees[tid][v - 1]++;
                if (force_symmetric && u != v) {
                    if (u > 0 && u <= num_vertices)
                        local_degrees[tid][u - 1]++;
                }
            } else {
                if (u > 0 && u <= num_vertices)
                    local_degrees[tid][u - 1]++;
                if (force_symmetric && u != v) {
                    if (v > 0 && v <= num_vertices)
                        local_degrees[tid][v - 1]++;
                }
            }
        }
    };

    const char* data_ptr = header.data_start;
    size_t data_size     = (size_t)(end - data_ptr);
    for (unsigned int i = 0; i < num_threads; ++i) {
        const char* c_start = data_ptr + (i * data_size / num_threads);
        const char* c_end   = data_ptr + ((i + 1) * data_size / num_threads);

        if (i > 0)
            c_start = next_line(c_start, end);
        if (i < num_threads - 1)
            c_end = next_line(c_end, end);
        else
            c_end = end;

        if (c_start < c_end)
            threads.emplace_back(process_chunk_count, c_start, c_end, i);
    }

    for (auto& t : threads)
        t.join();
    threads.clear();

    // Allocate host pinned memory for row offsets
    g.h_row_offsets =
        ctx.pinned_allocator->allocate((num_vertices + 1) * sizeof(uint64_t), alignof(uint64_t));
    g.row_offsets =
        std::span<uint64_t>(static_cast<uint64_t*>(g.h_row_offsets.mappedPtr), num_vertices + 1);

    // Calculate prefix sums and thread-specific starting positions
    std::vector<std::vector<uint64_t>> thread_write_positions(
        num_threads, std::vector<uint64_t>(num_vertices + 1, 0));

    uint64_t current_offset = 0;
    for (uint64_t i = 0; i < num_vertices; ++i) {
        g.row_offsets[i] = current_offset;
        for (unsigned int t = 0; t < num_threads; ++t) {
            thread_write_positions[t][i] = current_offset;
            current_offset += local_degrees[t][i];
        }
    }
    g.row_offsets[num_vertices] = current_offset;
    g.num_edges                 = current_offset;

    // Allocate host pinned memory for column indices
    g.h_column_indices =
        ctx.pinned_allocator->allocate(g.num_edges * sizeof(uint32_t), alignof(uint32_t));
    g.column_indices =
        std::span<uint32_t>(static_cast<uint32_t*>(g.h_column_indices.mappedPtr), g.num_edges);


    auto process_chunk_fill = [&](const char* c_start, const char* c_end, unsigned int tid) {
        const char* ptr = c_start;
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

            if (target_layout == GraphLayout::CSC) {
                if (v > 0 && v <= num_vertices) {
                    uint64_t pos          = thread_write_positions[tid][v - 1]++;
                    g.column_indices[pos] = (uint32_t)(u - 1);
                }
                if (force_symmetric && u != v) {
                    if (u > 0 && u <= num_vertices) {
                        uint64_t pos          = thread_write_positions[tid][u - 1]++;
                        g.column_indices[pos] = (uint32_t)(v - 1);
                    }
                }
            } else {
                if (u > 0 && u <= num_vertices) {
                    uint64_t pos          = thread_write_positions[tid][u - 1]++;
                    g.column_indices[pos] = (uint32_t)(v - 1);
                }
                if (force_symmetric && u != v) {
                    if (v > 0 && v <= num_vertices) {
                        uint64_t pos          = thread_write_positions[tid][v - 1]++;
                        g.column_indices[pos] = (uint32_t)(u - 1);
                    }
                }
            }
        }
    };

    for (unsigned int i = 0; i < num_threads; ++i) {
        const char* c_start = data_ptr + (i * data_size / num_threads);
        const char* c_end   = data_ptr + ((i + 1) * data_size / num_threads);

        if (i > 0)
            c_start = next_line(c_start, end);
        if (i < num_threads - 1)
            c_end = next_line(c_end, end);
        else
            c_end = end;

        if (c_start < c_end)
            threads.emplace_back(process_chunk_fill, c_start, c_end, i);
    }

    for (auto& t : threads)
        t.join();
        
    auto t_read = std::chrono::high_resolution_clock::now();

    // Sort and unique each adjacency list in parallel
    std::vector<uint64_t> unique_degrees(num_vertices, 0);
    
    auto process_dedup = [&](uint64_t start_v, uint64_t end_v) {
        std::vector<uint32_t> scratch;
        for (uint64_t v = start_v; v < end_v; ++v) {
            uint64_t start_edge = g.row_offsets[v];
            uint64_t end_edge = g.row_offsets[v + 1];
            if (end_edge > start_edge) {
                size_t len = end_edge - start_edge;
                scratch.resize(len);
                // Copy to cached RAM for fast random access
                std::copy(g.column_indices.begin() + start_edge, 
                          g.column_indices.begin() + end_edge, 
                          scratch.begin());
                          
                std::sort(scratch.begin(), scratch.end());
                auto last = std::unique(scratch.begin(), scratch.end());
                size_t unique_len = std::distance(scratch.begin(), last);
                unique_degrees[v] = unique_len;
                
                // Write back unique elements sequentially
                std::copy(scratch.begin(), scratch.begin() + unique_len, 
                          g.column_indices.begin() + start_edge);
            }
        }
    };

    std::vector<uint64_t> thread_v_starts(num_threads + 1, 0);
    thread_v_starts[num_threads] = num_vertices;
    uint64_t edges_per_thread = g.num_edges / num_threads;
    uint64_t current_acc = 0;
    unsigned int current_thread = 1;
    for (uint64_t i = 0; i < num_vertices; ++i) {
        current_acc += g.row_offsets[i+1] - g.row_offsets[i];
        if (current_acc >= current_thread * edges_per_thread && current_thread < num_threads) {
            thread_v_starts[current_thread] = i + 1;
            current_thread++;
        }
    }

    threads.clear();
    for (unsigned int i = 0; i < num_threads; ++i) {
        uint64_t start = thread_v_starts[i];
        uint64_t end = thread_v_starts[i+1];
        if (start < end)
            threads.emplace_back(process_dedup, start, end);
    }
    for (auto& t : threads) t.join();
    
    // Prefix sum the new true degrees
    std::vector<uint64_t> old_row_offsets(g.row_offsets.begin(), g.row_offsets.end());
    uint64_t true_edges = 0;
    for (uint64_t i = 0; i < num_vertices; ++i) {
        g.row_offsets[i] = true_edges;
        true_edges += unique_degrees[i];
    }
    g.row_offsets[num_vertices] = true_edges;
    g.num_edges = true_edges;
    g.column_indices = std::span<uint32_t>(static_cast<uint32_t*>(g.h_column_indices.mappedPtr), g.num_edges);

    // Shift sequentially to prevent data races
    for (uint64_t v = 0; v < num_vertices; ++v) {
        uint64_t old_start = old_row_offsets[v];
        uint64_t new_start = g.row_offsets[v];
        uint64_t count = unique_degrees[v];
        if (old_start != new_start && count > 0) {
            // Safe to copy because new_start <= old_start
            // and we are iterating sequentially (left to right)
            std::copy(g.column_indices.begin() + old_start, 
                      g.column_indices.begin() + old_start + count, 
                      g.column_indices.begin() + new_start);
        }
    }
    
    // Shift complete.
    
    auto t_topology = std::chrono::high_resolution_clock::now();

    // Upload to device
    //
    uint64_t row_offset_n        = (g.num_vertices + 1);
    uint64_t row_offset_n_padded = row_offset_n;
    CheckAllocation(g.d_row_offsets = ctx.pool_allocator->allocate(
                        row_offset_n_padded * sizeof(uint64_t), alignof(uint64_t)));
    CheckAllocation(g.d_column_indices =
                        ctx.pool_allocator->allocate(g.h_column_indices.size, alignof(uint32_t)));

    StreamHandle stream = ctx.create_stream();
    ctx.copy(g.h_row_offsets, g.d_row_offsets, stream);
    ctx.copy(g.h_column_indices, g.d_column_indices, stream);

    uint64_t toFill = (row_offset_n_padded - row_offset_n) * sizeof(uint64_t);
    if (toFill > 0) {
        ctx.fill(g.d_row_offsets, row_offset_n * sizeof(uint64_t), toFill,
                 std::numeric_limits<uint32_t>::max(), stream);
    }
    ctx.submit(stream);
    ctx.sync(stream);

    return g;
}
} // namespace graphs
} // namespace pictor
