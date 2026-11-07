/* SPDX-License-Identifier: MPL-2.0 */
/* Author: Himanshu Gupta <himanshu3@iisc.ac.in> */

#pragma once

#include "base_executor.hpp"

#include <array>

namespace pictor {

template <typename TCode, typename TParams, typename TResources = void>
class MeshExecutor : public BaseExecutor<TParams, TResources> {
  public:
    const uint32_t taskWgX;
    const uint32_t taskWgY;
    const uint32_t taskWgZ;
    const uint32_t meshWgX;
    const uint32_t meshWgY;
    const uint32_t meshWgZ;

    static Expected<std::unique_ptr<MeshExecutor<TCode, TParams, TResources>>, vk::Result>
    create(Context& ctx, const vk::SpecializationInfo* specInfo = nullptr,
           CompileOptions options = {})
    {

        auto taskRes = ctx.shader_compiler->compile(TCode::sources, std::string(TCode::task_entry),
                                                    ShaderStage::Task, ctx.subgroup_size, options);
        auto meshRes = ctx.shader_compiler->compile(TCode::sources, std::string(TCode::mesh_entry),
                                                    ShaderStage::Mesh, ctx.subgroup_size, options);

        if (!taskRes.success || !meshRes.success) {
            return vk::Result::eErrorInitializationFailed;
        }

        return create_from_spirv(ctx, taskRes.spirv, meshRes.spirv, std::string(TCode::task_entry),
                                 std::string(TCode::mesh_entry), specInfo, taskRes.threadGroupSize,
                                 meshRes.threadGroupSize);
    }

    // Tuned creation: `tuner` adjusts the shader's declared tunables for this device before the
    // modules are compiled. `TunerFor` and `tuned_defines` own the callable shape, the
    // device-limits contract and the shared step.
    template <typename TTuner>
        requires TunerFor<TCode, TTuner>
    static Expected<std::unique_ptr<MeshExecutor<TCode, TParams, TResources>>, vk::Result>
    create(Context& ctx, TTuner&& tuner, const vk::SpecializationInfo* specInfo = nullptr)
    {
        auto defines = tuned_defines<TCode>(ctx.device_info, std::forward<TTuner>(tuner));
        return create(ctx, specInfo, CompileOptions{defines.view()});
    }

    static Expected<std::unique_ptr<MeshExecutor<TCode, TParams, TResources>>, vk::Result>
    create_from_spirv(Context& ctx, const std::vector<uint32_t>& taskShader,
                      const std::vector<uint32_t>& meshShader,
                      const std::string& taskEntry = "main", const std::string& meshEntry = "main",
                      const vk::SpecializationInfo* specInfo      = nullptr,
                      std::array<uint32_t, 3> taskThreadGroupSize = {1, 1, 1},
                      std::array<uint32_t, 3> meshThreadGroupSize = {1, 1, 1})
    {
        auto executor = std::unique_ptr<MeshExecutor<TCode, TParams, TResources>>(
            new MeshExecutor<TCode, TParams, TResources>(
                taskThreadGroupSize[0], taskThreadGroupSize[1], taskThreadGroupSize[2],
                meshThreadGroupSize[0], meshThreadGroupSize[1], meshThreadGroupSize[2]));
        executor->ctx = &ctx;

        vk::ShaderStageFlags pushRangeStages =
            vk::ShaderStageFlagBits::eTaskEXT | vk::ShaderStageFlagBits::eMeshEXT;

        // Auto-Binding: Get bindings from the Resources struct if it exists
        std::vector<vk::DescriptorSetLayoutBinding> bindings;
        if constexpr (!std::is_void_v<TResources>) {
            bindings = TResources::get_bindings();
        }

        auto layoutRes = executor->_create_layout(pushRangeStages, bindings);
        if (layoutRes != vk::Result::eSuccess)
            return layoutRes;

        auto taskModuleRes = executor->_create_module(taskShader);
        if (taskModuleRes.result != vk::Result::eSuccess)
            return taskModuleRes.result;
        vk::ShaderModule taskModule = taskModuleRes.value;

        auto meshModuleRes = executor->_create_module(meshShader);
        if (meshModuleRes.result != vk::Result::eSuccess)
            return meshModuleRes.result;
        vk::ShaderModule meshModule = meshModuleRes.value;

        std::vector<vk::PipelineShaderStageCreateInfo> stages;
        stages.push_back({{}, vk::ShaderStageFlagBits::eTaskEXT, taskModule, "main", specInfo});
        stages.push_back({{}, vk::ShaderStageFlagBits::eMeshEXT, meshModule, "main", specInfo});

        vk::PipelineRasterizationStateCreateInfo rasterInfo{};
        rasterInfo.rasterizerDiscardEnable = VK_TRUE;
        rasterInfo.lineWidth               = 1.0f;

        vk::PipelineMultisampleStateCreateInfo msInfo{};
        msInfo.rasterizationSamples = vk::SampleCountFlagBits::e1;

        vk::PipelineRenderingCreateInfo renderingInfo{};

        vk::GraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.pNext               = &renderingInfo;
        pipelineInfo.stageCount          = static_cast<uint32_t>(stages.size());
        pipelineInfo.pStages             = stages.data();
        pipelineInfo.layout              = executor->layout;
        pipelineInfo.pRasterizationState = &rasterInfo;
        pipelineInfo.pMultisampleState   = &msInfo;

        auto pipeRes = executor->_device.createGraphicsPipeline(nullptr, pipelineInfo);

        if (taskModule)
            executor->_device.destroyShaderModule(taskModule);
        if (meshModule)
            executor->_device.destroyShaderModule(meshModule);

        if (pipeRes.result != vk::Result::eSuccess)
            return pipeRes.result;
        executor->pipeline    = pipeRes.value;
        executor->_entry_name = meshEntry;
        executor->ctx->set_name(executor->pipeline, meshEntry);

        return executor;
    }

    void record(uint32_t x, uint32_t y, uint32_t z, const TParams& pushConstants,
                StreamHandle handle = cDefaultStream)
    {
        Stream* stream = this->ctx->get_stream(handle);
        if (!stream)
            return;

        uint32_t x_ = (x + taskWgX - 1) / taskWgX;
        uint32_t y_ = (y + taskWgY - 1) / taskWgY;
        uint32_t z_ = (z + taskWgZ - 1) / taskWgZ;

        DirectParams params = {this, pushConstants, x_, y_, z_};
        stream->record(this->_entry_name.c_str(), _direct_record_cb, &params);
    }

    void record_raw(uint32_t x, uint32_t y, uint32_t z, const TParams& pushConstants,
                    StreamHandle handle = cDefaultStream)
    {
        Stream* stream = this->ctx->get_stream(handle);
        if (!stream)
            return;
        DirectParams params = {this, pushConstants, x, y, z};
        stream->record(this->_entry_name.c_str(), _direct_record_cb, &params);
    }

    void record_indirect(const AllocationInfo& indirectBuffer, const TParams& pushConstants,
                         StreamHandle handle = cDefaultStream)
    {
        Stream* stream = this->ctx->get_stream(handle);
        if (!stream)
            return;

        auto params =
            IndirectParams{.self = this, .pushConstants = pushConstants, .alloc = indirectBuffer};
        stream->record(this->_entry_name.c_str(), _indirect_record_cb, &params);
    }

  private:
    MeshExecutor() = default;
    MeshExecutor(uint32_t taskWgX, uint32_t taskWgY, uint32_t taskWgZ, uint32_t meshWgX,
                 uint32_t meshWgY, uint32_t meshWgZ)
        : taskWgX(taskWgX), taskWgY(taskWgY), taskWgZ(taskWgZ), meshWgX(meshWgX), meshWgY(meshWgY),
          meshWgZ(meshWgZ)
    {
    }

    struct DirectParams {
        MeshExecutor<TCode, TParams, TResources>* self;
        TParams const& pushConstants;
        uint32_t x, y, z;
    };

    struct IndirectParams {
        MeshExecutor<TCode, TParams, TResources>* self;
        TParams const& pushConstants;
        AllocationInfo const& alloc;
    };

    static void _direct_record_cb(vk::CommandBuffer cmd, void* data)
    {
        auto p = static_cast<DirectParams*>(data);

        vk::RenderingInfo renderingInfo{};
        renderingInfo.renderArea = vk::Rect2D{{0, 0}, {1, 1}};
        renderingInfo.layerCount = 1;

        cmd.beginRendering(renderingInfo);
        p->self->__bind_push_constants(cmd, vk::PipelineBindPoint::eGraphics, p->pushConstants);
        cmd.drawMeshTasksEXT(p->x, p->y, p->z);
        cmd.endRendering();
    }

    static void _indirect_record_cb(vk::CommandBuffer cmd, void* data)
    {
        auto p = static_cast<IndirectParams*>(data);

        vk::RenderingInfo renderingInfo{};
        renderingInfo.renderArea = vk::Rect2D{{0, 0}, {1, 1}};
        renderingInfo.layerCount = 1;

        cmd.beginRendering(renderingInfo);
        p->self->__bind_push_constants(cmd, vk::PipelineBindPoint::eGraphics, p->pushConstants);
        cmd.drawMeshTasksIndirectEXT(p->alloc.buffer, p->alloc.offset, 1, 0);
        cmd.endRendering();
    }
};

} // namespace pictor
