// SPDX-License-Identifier: GPL-2.0-only
// Author: Himanshu Gupta <himanshu3@iisc.ac.in>

#include "host_graph.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace pictor {
namespace graphs {

namespace {

struct MappedFile {
    void*  data = nullptr;
    size_t size = 0;

    explicit MappedFile(const std::string& filename)
    {
#ifdef _WIN32
        HANDLE file = CreateFileA(filename.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                                  OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file == INVALID_HANDLE_VALUE)
            throw std::runtime_error("Could not open file: " + filename);

        LARGE_INTEGER file_size{};
        if (!GetFileSizeEx(file, &file_size) || file_size.QuadPart <= 0 ||
            static_cast<unsigned long long>(file_size.QuadPart) >
                std::numeric_limits<size_t>::max()) {
            CloseHandle(file);
            throw std::runtime_error("Could not stat file: " + filename);
        }
        size = static_cast<size_t>(file_size.QuadPart);

        HANDLE mapping = CreateFileMappingA(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
        CloseHandle(file);
        if (!mapping)
            throw std::runtime_error("Could not create file mapping: " + filename);

        data = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
        CloseHandle(mapping);
        if (!data)
            throw std::runtime_error("Could not map file: " + filename);
#else
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
#endif
    }

    ~MappedFile()
    {
#ifdef _WIN32
        if (data)
            UnmapViewOfFile(data);
#else
        if (data && data != MAP_FAILED)
            munmap(data, size);
#endif
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

constexpr uint32_t kCacheMagic   = 0x4D435331; // 'MCS1'
constexpr uint32_t kCacheVersion = 2;

std::string cache_path(const std::string& file, bool sym, GraphLayout layout)
{
    const char* suffix = (layout == GraphLayout::CSC) ? ".cscbin" : ".csrbin";
    return sym ? file + ".sym" + suffix : file + suffix;
}

} // namespace

void save_binary(const HostGraph& graph, const std::string& path)
{
    const std::string tmp = path + ".tmp";
    FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f)
        throw std::runtime_error("Could not write cache: " + tmp);

    const uint32_t magic = kCacheMagic, version = kCacheVersion,
                   sym = graph.symmetrized ? 1u : 0u;
    const bool ok =
        std::fwrite(&magic, 4, 1, f) == 1 && std::fwrite(&version, 4, 1, f) == 1 &&
        std::fwrite(&sym, 4, 1, f) == 1 && std::fwrite(&graph.num_vertices, 8, 1, f) == 1 &&
        std::fwrite(&graph.num_edges, 8, 1, f) == 1 &&
        std::fwrite(graph.row_offsets.data(), 8, graph.num_vertices + 1, f) ==
            graph.num_vertices + 1 &&
        std::fwrite(graph.column_indices.data(), 4, graph.num_edges, f) == graph.num_edges;
    std::fclose(f);
    if (!ok) {
        std::remove(tmp.c_str());
        throw std::runtime_error("Could not write cache: " + tmp);
    }
    std::rename(tmp.c_str(), path.c_str());
}

HostGraph load_binary(const std::string& path)
{
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f)
        throw std::runtime_error("Could not open cache: " + path);

    uint32_t magic = 0, version = 0, sym = 0;
    uint64_t n = 0, m = 0;
    bool ok = std::fread(&magic, 4, 1, f) == 1 && magic == kCacheMagic &&
              std::fread(&version, 4, 1, f) == 1 && version == kCacheVersion &&
              std::fread(&sym, 4, 1, f) == 1 && std::fread(&n, 8, 1, f) == 1 &&
              std::fread(&m, 8, 1, f) == 1;

    HostGraph g;
    if (ok) {
        g.num_vertices = n;
        g.num_edges    = m;
        g.symmetrized  = sym != 0;
        g.row_offsets.resize(n + 1);
        g.column_indices.resize(m);
        ok = std::fread(g.row_offsets.data(), 8, n + 1, f) == n + 1 &&
             std::fread(g.column_indices.data(), 4, m, f) == m;
    }
    std::fclose(f);
    if (!ok)
        throw std::runtime_error("Invalid cache: " + path);
    return g;
}

HostGraph import_mtx(const std::string& file, bool symmetrize, GraphLayout layout, bool use_cache)
{
    MappedFile mf(file);
    const char* start = static_cast<const char*>(mf.data);
    const char* end   = start + mf.size;

    const MtxHeader header    = parse_header(start, end);
    const bool force_symmetric = symmetrize || header.symmetric;
    const uint64_t n           = std::max(header.rows, header.cols);

    if (use_cache) {
        try {
            HostGraph cached = load_binary(cache_path(file, force_symmetric, layout));
            if (cached.symmetrized == force_symmetric)
                return cached;
        } catch (...) {
            // fall through to a full parse
        }
    }

    HostGraph g;
    g.num_vertices = n;
    g.layout       = layout;
    g.symmetrized  = force_symmetric;

    unsigned num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
        num_threads = 1;

    // Pass 1: per-thread per-vertex degrees (no atomics).
    std::vector<std::vector<uint64_t>> degrees(num_threads, std::vector<uint64_t>(n + 1, 0));
    const char*  data_ptr  = header.data_start;
    const size_t data_size = (size_t)(end - data_ptr);

    auto count_chunk = [&](const char* c_start, const char* c_end, unsigned tid) {
        const char* ptr = c_start;
        while (ptr < c_end) {
            uint64_t u = 0, v = 0;
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

            if (layout == GraphLayout::CSC) {
                if (v > 0 && v <= n)
                    degrees[tid][v - 1]++;
                if (force_symmetric && u != v && u > 0 && u <= n)
                    degrees[tid][u - 1]++;
            } else {
                if (u > 0 && u <= n)
                    degrees[tid][u - 1]++;
                if (force_symmetric && u != v && v > 0 && v <= n)
                    degrees[tid][v - 1]++;
            }
        }
    };

    std::vector<std::thread> threads;
    for (unsigned i = 0; i < num_threads; ++i) {
        const char* c_start = data_ptr + (i * data_size / num_threads);
        const char* c_end   = data_ptr + ((i + 1) * data_size / num_threads);
        if (i > 0)
            c_start = next_line(c_start, end);
        if (i < num_threads - 1)
            c_end = next_line(c_end, end);
        if (c_start < c_end)
            threads.emplace_back(count_chunk, c_start, c_end, i);
    }
    for (auto& t : threads)
        t.join();

    // Prefix sums; the per-thread degree arrays are rewritten in place into
    // per-thread write positions.
    g.row_offsets.resize(n + 1);
    uint64_t current_offset = 0;
    for (uint64_t i = 0; i < n; ++i) {
        g.row_offsets[i] = current_offset;
        for (unsigned t = 0; t < num_threads; ++t) {
            const uint64_t d = degrees[t][i];
            degrees[t][i]    = current_offset;
            current_offset += d;
        }
    }
    g.row_offsets[n] = current_offset;
    g.num_edges      = current_offset;
    g.column_indices.resize(current_offset);

    // Pass 2: fill the adjacency lists.
    auto fill_chunk = [&](const char* c_start, const char* c_end, unsigned tid) {
        const char* ptr = c_start;
        while (ptr < c_end) {
            uint64_t u = 0, v = 0;
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

            if (layout == GraphLayout::CSC) {
                if (v > 0 && v <= n)
                    g.column_indices[degrees[tid][v - 1]++] = (uint32_t)(u - 1);
                if (force_symmetric && u != v && u > 0 && u <= n)
                    g.column_indices[degrees[tid][u - 1]++] = (uint32_t)(v - 1);
            } else {
                if (u > 0 && u <= n)
                    g.column_indices[degrees[tid][u - 1]++] = (uint32_t)(v - 1);
                if (force_symmetric && u != v && v > 0 && v <= n)
                    g.column_indices[degrees[tid][v - 1]++] = (uint32_t)(u - 1);
            }
        }
    };

    threads.clear();
    for (unsigned i = 0; i < num_threads; ++i) {
        const char* c_start = data_ptr + (i * data_size / num_threads);
        const char* c_end   = data_ptr + ((i + 1) * data_size / num_threads);
        if (i > 0)
            c_start = next_line(c_start, end);
        if (i < num_threads - 1)
            c_end = next_line(c_end, end);
        if (c_start < c_end)
            threads.emplace_back(fill_chunk, c_start, c_end, i);
    }
    for (auto& t : threads)
        t.join();

    // Sort + deduplicate each adjacency list, compacting into new arrays.
    std::vector<uint64_t> unique_len(n, 0);
    {
        std::vector<std::thread> workers;
        auto dedup_range = [&](uint64_t v_begin, uint64_t v_end) {
            std::vector<uint32_t> scratch;
            for (uint64_t v = v_begin; v < v_end; ++v) {
                const uint64_t b = g.row_offsets[v];
                const uint64_t e = g.row_offsets[v + 1];
                if (e > b) {
                    scratch.assign(g.column_indices.begin() + b, g.column_indices.begin() + e);
                    std::sort(scratch.begin(), scratch.end());
                    const auto last = std::unique(scratch.begin(), scratch.end());
                    unique_len[v]   = (uint64_t)(last - scratch.begin());
                    std::copy(scratch.begin(), last, g.column_indices.begin() + b);
                }
            }
        };
        for (unsigned t = 0; t < num_threads; ++t)
            workers.emplace_back(dedup_range, n * t / num_threads, n * (t + 1) / num_threads);
        for (auto& t : workers)
            t.join();
    }

    std::vector<uint64_t> new_offsets(n + 1, 0);
    uint64_t              acc = 0;
    for (uint64_t i = 0; i < n; ++i) {
        new_offsets[i] = acc;
        acc += unique_len[i];
    }
    new_offsets[n] = acc;

    std::vector<uint32_t> new_columns(acc);
    {
        std::vector<std::thread> workers;
        auto copy_range = [&](uint64_t v_begin, uint64_t v_end) {
            for (uint64_t v = v_begin; v < v_end; ++v) {
                const uint64_t len = new_offsets[v + 1] - new_offsets[v];
                if (len)
                    std::memcpy(new_columns.data() + new_offsets[v],
                                g.column_indices.data() + g.row_offsets[v],
                                len * sizeof(uint32_t));
            }
        };
        for (unsigned t = 0; t < num_threads; ++t)
            workers.emplace_back(copy_range, n * t / num_threads, n * (t + 1) / num_threads);
        for (auto& t : workers)
            t.join();
    }

    g.row_offsets.swap(new_offsets);
    g.column_indices.swap(new_columns);
    g.num_edges = acc;

    if (use_cache) {
        try {
            save_binary(g, cache_path(file, force_symmetric, layout));
        } catch (...) {
            // caching is best-effort
        }
    }
    return g;
}

} // namespace graphs
} // namespace pictor
