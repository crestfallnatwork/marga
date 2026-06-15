#include "pagerank.hpp"
#include "pictor_core/context/context.hpp"
#include "pictor_core/execution/shader_compiler.hpp"

// Generated headers from Slang
#include "out_degree.slang.h"
#include "pagerank.slang.h"

#include <iostream>
#include <cstring>
#include <chrono>

namespace pictor {
namespace graphs {

PageRank::PageRank(Context& ctx) : _ctx(ctx) {
    auto compiler_res = ShaderCompiler::create();
    if (!compiler_res.ok()) throw std::runtime_error("Failed to create shader compiler");
    auto& compiler = *compiler_res.val();

    // 1. Compile Out-Degree Shader
    std::string out_deg_source(reinterpret_cast<char*>(out_degree_slang), out_degree_slang_len);
    auto compute_res = compiler.compile(out_deg_source, "calc_out_degrees", ShaderStage::Compute);
    if (!compute_res.success) throw std::runtime_error("Out-degree compilation failed");
    
    auto compute_exec_res = ComputeExecutor<out_degree_json_layout::Params>::create_from_spirv(
        _ctx, compute_res.spirv, "main");
    if (!compute_exec_res.ok()) {
        std::cerr << "Failed to create out-degree executor. Vulkan result: " << vk::to_string(compute_exec_res.err()) << std::endl;
        throw std::runtime_error("Failed to create out-degree executor");
    }
    _out_degree_exec = std::move(compute_exec_res.val());

    // 2. Compile PageRank Pipeline Shaders
    std::string pr_source(reinterpret_cast<char*>(pagerank_slang), pagerank_slang_len);
    
    auto pr_task_res = compiler.compile(pr_source, "task", ShaderStage::Task);
    auto pr_mesh_res = compiler.compile(pr_source, "mesh", ShaderStage::Mesh);
    auto pr_reduce_res = compiler.compile(pr_source, "reduce", ShaderStage::Compute);
    auto pr_finalize_res = compiler.compile(pr_source, "finalize_and_check", ShaderStage::Compute);
    auto pr_dangling_res = compiler.compile(pr_source, "calc_dangling", ShaderStage::Compute);

    if (!pr_task_res.success || !pr_mesh_res.success || !pr_reduce_res.success || !pr_finalize_res.success || !pr_dangling_res.success) {
        std::cerr << "PageRank compilation failed" << std::endl;
        throw std::runtime_error("PageRank compilation failed");
    }

    auto pr_mesh_exec_res = MeshExecutor<pagerank_json_layout::Params>::create_from_spirv(
        _ctx, pr_task_res.spirv, pr_mesh_res.spirv, "main", "main");
    if (!pr_mesh_exec_res.ok()) throw std::runtime_error("Failed to create PR mesh executor");
    _pr_mesh_exec = std::move(pr_mesh_exec_res.val());

    auto pr_reduce_exec_res = ComputeExecutor<pagerank_json_layout::Params>::create_from_spirv(
        _ctx, pr_reduce_res.spirv, "main");
    if (!pr_reduce_exec_res.ok()) throw std::runtime_error("Failed to create PR reduce executor");
    _pr_reduce_exec = std::move(pr_reduce_exec_res.val());

    auto pr_finalize_exec_res = ComputeExecutor<pagerank_json_layout::Params>::create_from_spirv(
        _ctx, pr_finalize_res.spirv, "main");
    if (!pr_finalize_exec_res.ok()) throw std::runtime_error("Failed to create PR finalize executor");
    _pr_finalize_exec = std::move(pr_finalize_exec_res.val());

    auto pr_dangling_exec_res = ComputeExecutor<pagerank_json_layout::Params>::create_from_spirv(
        _ctx, pr_dangling_res.spirv, "main");
    if (!pr_dangling_exec_res.ok()) throw std::runtime_error("Failed to create PR dangling executor");
    _pr_dangling_exec = std::move(pr_dangling_exec_res.val());
}

PageRank::~PageRank() {
    _cleanup_buffers();
}

void PageRank::_allocate_buffers(uint64_t num_vertices, uint64_t num_edges) {
    _cleanup_buffers();
    _out_degrees = _ctx.pool_allocator->allocate(num_vertices * sizeof(uint32_t), alignof(uint32_t));
    _pr_current = _ctx.pool_allocator->allocate(num_vertices * sizeof(uint64_t), alignof(uint64_t));
    _pr_next = _ctx.pool_allocator->allocate(num_vertices * sizeof(uint64_t), alignof(uint64_t));
    _active_vertices_curr = _ctx.pool_allocator->allocate(num_vertices * sizeof(uint32_t), alignof(uint32_t));
    _active_count_curr = _ctx.pool_allocator->allocate(sizeof(uint32_t), alignof(uint32_t));
    _dangling_sum = _ctx.pool_allocator->allocate(sizeof(uint64_t), alignof(uint64_t));
    
    // Intermediate array needs to hold up to 32 partial sums per mesh workgroup.
    // Number of mesh workgroups is roughly num_edges / 128 + num_vertices
    // We size it generously. A single PartialSum is 12 bytes.
    uint64_t max_mesh_wgs = (num_edges / 128) + num_vertices;
    uint64_t intermediate_slots = max_mesh_wgs * 32;
    _intermediate = _ctx.pool_allocator->allocate(intermediate_slots * 12, alignof(uint32_t));
    _intermediate_count = _ctx.pool_allocator->allocate(sizeof(uint32_t), alignof(uint32_t));
    
    _h_pr_results = _ctx.pinned_allocator->allocate(num_vertices * sizeof(uint64_t), alignof(uint64_t));
}

void PageRank::_cleanup_buffers() {
    if (_out_degrees.allocation) _ctx.deallocate(_out_degrees);
    if (_pr_current.allocation) _ctx.deallocate(_pr_current);
    if (_pr_next.allocation) _ctx.deallocate(_pr_next);
    if (_active_vertices_curr.allocation) _ctx.deallocate(_active_vertices_curr);
    if (_active_count_curr.allocation) _ctx.deallocate(_active_count_curr);
    if (_dangling_sum.allocation) _ctx.deallocate(_dangling_sum);
    if (_intermediate.allocation) _ctx.deallocate(_intermediate);
    if (_intermediate_count.allocation) _ctx.deallocate(_intermediate_count);
    if (_h_pr_results.allocation) _ctx.deallocate(_h_pr_results);
}

void PageRank::preprocess(const Graph& graph) {
    if (graph.layout != GraphLayout::CSC) {
        throw std::runtime_error("PageRank requires a CSC graph for efficient execution");
    }

    if (_out_degrees.size != graph.num_vertices * sizeof(uint32_t)) {
        _allocate_buffers(graph.num_vertices, graph.num_edges);
    }

    StreamHandle stream = cDefaultStream;

    auto t_init_start = std::chrono::high_resolution_clock::now();
    _ctx.fill(_out_degrees, 0, _out_degrees.size, 0, stream);
    _ctx.barrier(_out_degrees, Access::Write, Access::Write, stream);
    _ctx.flush(stream);
    _ctx.sync(stream);
    auto t_init_end = std::chrono::high_resolution_clock::now();

    auto t_compute_start = std::chrono::high_resolution_clock::now();
    out_degree_json_layout::Params params;
    params.row_indices = graph.d_column_indices.address;
    params.out_degrees = _out_degrees.address;
    params.num_edges = graph.num_edges;

    uint32_t num_groups = (graph.num_edges + 255) / 256;
    _out_degree_exec->record(num_groups, 1, 1, params, stream);

    _ctx.barrier(_out_degrees, Access::Write, Access::Read, stream);
    
    _ctx.flush(stream);
    _ctx.sync(stream);

    auto t_compute_end = std::chrono::high_resolution_clock::now();
    
    double init_ms = std::chrono::duration<double, std::milli>(t_init_end - t_init_start).count();
    double compute_ms = std::chrono::duration<double, std::milli>(t_compute_end - t_compute_start).count();
    
    std::cout << "--- PageRank Preprocessing (Out-Degree Calculation) ---" << std::endl;
    std::cout << "  Init/Zero Time: " << init_ms << " ms" << std::endl;
    std::cout << "  Compute Time:   " << compute_ms << " ms" << std::endl;
}

std::pair<std::vector<float>, PageRankTimings> PageRank::run(const Graph& graph, uint32_t max_iterations) {
    PageRankTimings timings;
    auto t_full_start = std::chrono::high_resolution_clock::now();
    
    preprocess(graph);
    
    auto t_prep_end = std::chrono::high_resolution_clock::now();
    timings.preprocessing_ms = std::chrono::duration<double, std::milli>(t_prep_end - t_full_start).count();

    StreamHandle stream = cDefaultStream;

    // Initialization Phase
    auto t_init_start = std::chrono::high_resolution_clock::now();
    
    // Fixed point representation: PR = 1.0/N
    uint64_t fixed_scale = 1125899906842624ull; // 2^50
    uint64_t initial_pr = fixed_scale / graph.num_vertices;
    
    _ctx.fill(_pr_next, 0, _pr_next.size, 0, stream);
    std::vector<uint64_t> init_ranks(graph.num_vertices, initial_pr);
    _ctx.update(_pr_current, 0, graph.num_vertices * sizeof(uint64_t), init_ranks.data(), stream);

    std::vector<uint32_t> init_active(graph.num_vertices);
    for (uint32_t i=0; i<graph.num_vertices; i++) init_active[i] = i;
    _ctx.update(_active_vertices_curr, 0, graph.num_vertices * sizeof(uint32_t), init_active.data(), stream);
    
    uint32_t active_count = graph.num_vertices;
    _ctx.update(_active_count_curr, 0, sizeof(uint32_t), &active_count, stream);

    _ctx.flush(stream);
    _ctx.sync(stream);
    
    auto t_init_end = std::chrono::high_resolution_clock::now();
    timings.init_ms = std::chrono::duration<double, std::milli>(t_init_end - t_init_start).count();

    pagerank_json_layout::Params params;
    params.row_offsets = graph.d_row_offsets.address;
    params.column_indices = graph.d_column_indices.address;
    params.out_degrees = _out_degrees.address;
    params.pr_current = _pr_current.address;
    params.pr_next = _pr_next.address;
    params.active_vertices_curr = _active_vertices_curr.address;
    params.active_count_curr = _active_count_curr.address;
    params.intermediate = _intermediate.address;
    params.intermediate_count = _intermediate_count.address;
    params.dangling_sum = _dangling_sum.address;
    params.num_vertices = graph.num_vertices;
    
    uint64_t damping_num = 85;
    uint64_t damping_den = 100;
    
    uint64_t base_rank = (fixed_scale * (damping_den - damping_num)) / (damping_den * graph.num_vertices);
    
    params.damping_numerator = (uint32_t)damping_num;
    params.damping_denominator = (uint32_t)damping_den;
    
    params.base_rank_low = (uint32_t)(base_rank & 0xFFFFFFFF);
    params.base_rank_high = (uint32_t)(base_rank >> 32);

    auto t_iter_start = std::chrono::high_resolution_clock::now();
    
    for (uint32_t iter = 0; iter < max_iterations; ++iter) {
        struct LoopParams {
            PageRank* self;
            pagerank_json_layout::Params params;
            uint32_t dang_groups;
            uint32_t task_groups;
            uint32_t fin_groups;
        };

        LoopParams* lp = new LoopParams{
            this, 
            params, 
            (uint32_t)(graph.num_vertices + 255) / 256,
            (uint32_t)(graph.num_vertices + 31) / 32,
            (uint32_t)(graph.num_vertices + 255) / 256
        };

        _ctx.get_stream(stream)->record([](vk::CommandBuffer cmd, void* data) {
            auto* p = static_cast<LoopParams*>(data);
            
            // 0. Reset intermediate count
            cmd.fillBuffer(p->self->_intermediate_count.buffer, p->self->_intermediate_count.offset, p->self->_intermediate_count.size, 0);

            // 1. Calculate Dangling Sum
            cmd.fillBuffer(p->self->_dangling_sum.buffer, p->self->_dangling_sum.offset, p->self->_dangling_sum.size, 0);
            
            vk::BufferMemoryBarrier2 fillBarriers[2];
            fillBarriers[0].srcStageMask = vk::PipelineStageFlagBits2::eTransfer;
            fillBarriers[0].srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
            fillBarriers[0].dstStageMask = vk::PipelineStageFlagBits2::eComputeShader;
            fillBarriers[0].dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite;
            fillBarriers[0].buffer = p->self->_dangling_sum.buffer;
            fillBarriers[0].offset = p->self->_dangling_sum.offset;
            fillBarriers[0].size = p->self->_dangling_sum.size;

            fillBarriers[1].srcStageMask = vk::PipelineStageFlagBits2::eTransfer;
            fillBarriers[1].srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
            fillBarriers[1].dstStageMask = vk::PipelineStageFlagBits2::eTaskShaderEXT;
            fillBarriers[1].dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite;
            fillBarriers[1].buffer = p->self->_intermediate_count.buffer;
            fillBarriers[1].offset = p->self->_intermediate_count.offset;
            fillBarriers[1].size = p->self->_intermediate_count.size;

            vk::DependencyInfo fillDep{};
            fillDep.bufferMemoryBarrierCount = 2;
            fillDep.pBufferMemoryBarriers = fillBarriers;
            cmd.pipelineBarrier2(fillDep);

            p->self->_pr_dangling_exec->__execute_bda(cmd, p->dang_groups, 1, 1, p->params);

            vk::BufferMemoryBarrier2 dangBarrier{};
            dangBarrier.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader;
            dangBarrier.srcAccessMask = vk::AccessFlagBits2::eShaderWrite;
            dangBarrier.dstStageMask = vk::PipelineStageFlagBits2::eComputeShader;
            dangBarrier.dstAccessMask = vk::AccessFlagBits2::eShaderRead;
            dangBarrier.buffer = p->self->_dangling_sum.buffer;
            dangBarrier.offset = p->self->_dangling_sum.offset;
            dangBarrier.size = p->self->_dangling_sum.size;
            
            vk::DependencyInfo dangDep{};
            dangDep.bufferMemoryBarrierCount = 1;
            dangDep.pBufferMemoryBarriers = &dangBarrier;
            cmd.pipelineBarrier2(dangDep);

            // 2. Mesh Pull & Scatter to Intermediate
            p->self->_pr_mesh_exec->__execute_bda(cmd, p->task_groups, 1, 1, p->params);

            vk::BufferMemoryBarrier2 meshBarriers[2];
            meshBarriers[0].srcStageMask = vk::PipelineStageFlagBits2::eMeshShaderEXT | vk::PipelineStageFlagBits2::eTaskShaderEXT;
            meshBarriers[0].srcAccessMask = vk::AccessFlagBits2::eShaderWrite;
            meshBarriers[0].dstStageMask = vk::PipelineStageFlagBits2::eComputeShader;
            meshBarriers[0].dstAccessMask = vk::AccessFlagBits2::eShaderRead;
            meshBarriers[0].buffer = p->self->_intermediate.buffer;
            meshBarriers[0].offset = p->self->_intermediate.offset;
            meshBarriers[0].size = p->self->_intermediate.size;

            meshBarriers[1].srcStageMask = vk::PipelineStageFlagBits2::eTaskShaderEXT;
            meshBarriers[1].srcAccessMask = vk::AccessFlagBits2::eShaderWrite;
            meshBarriers[1].dstStageMask = vk::PipelineStageFlagBits2::eComputeShader;
            meshBarriers[1].dstAccessMask = vk::AccessFlagBits2::eShaderRead;
            meshBarriers[1].buffer = p->self->_intermediate_count.buffer;
            meshBarriers[1].offset = p->self->_intermediate_count.offset;
            meshBarriers[1].size = p->self->_intermediate_count.size;

            vk::DependencyInfo meshDep{};
            meshDep.bufferMemoryBarrierCount = 2;
            meshDep.pBufferMemoryBarriers = meshBarriers;
            cmd.pipelineBarrier2(meshDep);

            // 3. Reduce Intermediate
            // Since we don't know the exact count on the CPU, we do an indirect dispatch.
            // Wait, we don't have an indirect buffer setup for the compute shader here.
            // But we CAN over-dispatch based on max possible partials: (total_edges / 128 + num_vertices) * 32
            // And the compute shader just returns if threadId >= intermediate_count.
            uint32_t max_partials = (212698418 / 128 + 3000000) * 32; // Just a safe upper bound
            uint32_t reduce_groups = (max_partials + 255) / 256;
            p->self->_pr_reduce_exec->__execute_bda(cmd, reduce_groups, 1, 1, p->params);

            vk::BufferMemoryBarrier2 reduceBarrier{};
            reduceBarrier.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader;
            reduceBarrier.srcAccessMask = vk::AccessFlagBits2::eShaderWrite;
            reduceBarrier.dstStageMask = vk::PipelineStageFlagBits2::eComputeShader;
            reduceBarrier.dstAccessMask = vk::AccessFlagBits2::eShaderRead;
            reduceBarrier.buffer = p->self->_pr_next.buffer;
            reduceBarrier.offset = p->self->_pr_next.offset;
            reduceBarrier.size = p->self->_pr_next.size;

            vk::DependencyInfo reduceDep{};
            reduceDep.bufferMemoryBarrierCount = 1;
            reduceDep.pBufferMemoryBarriers = &reduceBarrier;
            cmd.pipelineBarrier2(reduceDep);

            // 4. Finalize & Check
            p->self->_pr_finalize_exec->__execute_bda(cmd, p->fin_groups, 1, 1, p->params);
            
            vk::BufferMemoryBarrier2 finBarrier{};
            finBarrier.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader;
            finBarrier.srcAccessMask = vk::AccessFlagBits2::eShaderWrite;
            finBarrier.dstStageMask = vk::PipelineStageFlagBits2::eMeshShaderEXT | vk::PipelineStageFlagBits2::eComputeShader;
            finBarrier.dstAccessMask = vk::AccessFlagBits2::eShaderRead;
            finBarrier.buffer = p->self->_pr_current.buffer;
            finBarrier.offset = p->self->_pr_current.offset;
            finBarrier.size = p->self->_pr_current.size;

            vk::DependencyInfo finDep{};
            finDep.bufferMemoryBarrierCount = 1;
            finDep.pBufferMemoryBarriers = &finBarrier;
            cmd.pipelineBarrier2(finDep);

            delete p;
        }, lp);

        _ctx.flush(stream);
        _ctx.sync(stream);
    }

    auto t_iter_end = std::chrono::high_resolution_clock::now();
    double total_iter_ms = std::chrono::duration<double, std::milli>(t_iter_end - t_iter_start).count();
    timings.iteration_avg_ms = total_iter_ms / max_iterations;
    timings.num_iterations = max_iterations;

    // Readback results
    _ctx.copy(_pr_current, _h_pr_results, stream);
    _ctx.flush(stream);
    _ctx.sync(stream);

    auto t_full_end = std::chrono::high_resolution_clock::now();
    timings.total_ms = std::chrono::duration<double, std::milli>(t_full_end - t_full_start).count();

    // Convert fixed-point back to float for user
    std::vector<float> final_pr(graph.num_vertices);
    uint64_t* raw_pr = static_cast<uint64_t*>(_h_pr_results.mappedPtr);
    for (uint32_t i=0; i<graph.num_vertices; i++) {
        final_pr[i] = static_cast<float>(raw_pr[i]) / (double)fixed_scale;
    }

    return {final_pr, timings};
}

} // namespace graphs
} // namespace pictor
