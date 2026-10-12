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

#include <cstring>

namespace pictor {
namespace graphs {

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

void Graph::save_to_binary(const std::string& path) const
{
    HostGraph host;
    host.num_vertices = num_vertices;
    host.num_edges    = num_edges;
    host.layout       = layout;
    host.symmetrized  = symmetrized;
    host.row_offsets.assign(row_offsets.begin(), row_offsets.end());
    host.column_indices.assign(column_indices.begin(), column_indices.end());
    save_binary(host, path);
}

void Graph::load_from_binary(const std::string& path)
{
    const HostGraph host = load_binary(path);
    num_vertices         = host.num_vertices;
    num_edges            = host.num_edges;
    layout               = host.layout;
    symmetrized          = host.symmetrized;
    upload_host_graph(host);
}

void Graph::upload_host_graph(const HostGraph& host)
{
    h_row_offsets = ctx->pinned_allocator->allocate((host.num_vertices + 1) * sizeof(uint64_t),
                                                    alignof(uint64_t));
    row_offsets   = std::span<uint64_t>(static_cast<uint64_t*>(h_row_offsets.mappedPtr),
                                        host.num_vertices + 1);
    std::memcpy(row_offsets.data(), host.row_offsets.data(),
                (host.num_vertices + 1) * sizeof(uint64_t));

    h_column_indices = ctx->pinned_allocator->allocate(host.num_edges * sizeof(uint32_t),
                                                       alignof(uint32_t));
    column_indices   = std::span<uint32_t>(static_cast<uint32_t*>(h_column_indices.mappedPtr),
                                           host.num_edges);
    std::memcpy(column_indices.data(), host.column_indices.data(),
                host.num_edges * sizeof(uint32_t));

    CheckAllocation(d_row_offsets =
                        ctx->pool_allocator->allocate((host.num_vertices + 1) * sizeof(uint64_t),
                                                      alignof(uint64_t)));
    CheckAllocation(d_column_indices =
                        ctx->pool_allocator->allocate(host.num_edges * sizeof(uint32_t),
                                                      alignof(uint32_t)));

    StreamHandle stream = ctx->create_stream();
    ctx->copy(h_row_offsets, d_row_offsets, stream);
    ctx->copy(h_column_indices, d_column_indices, stream);
    ctx->submit(stream);
    ctx->sync(stream);
}

Graph Graph::import(Context& ctx, std::string const& file, bool symmetrize,
                    GraphLayout target_layout)
{
    const HostGraph host = import_mtx(file, symmetrize, target_layout);

    Graph g(&ctx);
    g.num_vertices = host.num_vertices;
    g.num_edges    = host.num_edges;
    g.layout       = host.layout;
    g.symmetrized  = host.symmetrized;
    g.upload_host_graph(host);
    return g;
}

} // namespace graphs
} // namespace pictor
