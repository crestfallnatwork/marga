#pragma once

#include "base_executor.hpp"

namespace pictor {

template <typename TCode, typename TParams, typename TResources = void>
class MeshExecutor : public BaseExecutor<TParams, TResources> {
public:
  static Expected<std::unique_ptr<MeshExecutor<TCode, TParams, TResources>>,
                  vk::Result>
  create(Context &ctx,
         const vk::SpecializationInfo *specInfo = nullptr) {

    auto taskRes = ctx.shader_compiler->compile(std::string(TCode::source),
                                    std::string(TCode::task_entry),
                                    ShaderStage::Task);
    auto meshRes = ctx.shader_compiler->compile(std::string(TCode::source),
                                    std::string(TCode::mesh_entry),
                                    ShaderStage::Mesh);

    if (!taskRes.success || !meshRes.success) {
      return vk::Result::eErrorInitializationFailed;
    }

    return create_from_spirv(ctx, taskRes.spirv, meshRes.spirv,
                             std::string(TCode::task_entry),
                             std::string(TCode::mesh_entry), specInfo);
  }

  static Expected<std::unique_ptr<MeshExecutor<TCode, TParams, TResources>>,
                  vk::Result>
  create_from_spirv(Context &ctx, const std::vector<uint32_t> &taskShader,
                    const std::vector<uint32_t> &meshShader,
                    const std::string &taskEntry = "main",
                    const std::string &meshEntry = "main",
                    const vk::SpecializationInfo *specInfo = nullptr) {

    auto executor = std::unique_ptr<MeshExecutor<TCode, TParams, TResources>>(
        new MeshExecutor<TCode, TParams, TResources>());
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
    stages.push_back({{},
                      vk::ShaderStageFlagBits::eTaskEXT,
                      taskModule,
                      "main",
                      specInfo});
    stages.push_back({{},
                      vk::ShaderStageFlagBits::eMeshEXT,
                      meshModule,
                      "main",
                      specInfo});

    vk::PipelineRasterizationStateCreateInfo rasterInfo{};
    rasterInfo.rasterizerDiscardEnable = VK_TRUE;
    rasterInfo.lineWidth = 1.0f;

    vk::PipelineMultisampleStateCreateInfo msInfo{};
    msInfo.rasterizationSamples = vk::SampleCountFlagBits::e1;

    vk::PipelineRenderingCreateInfo renderingInfo{};

    vk::GraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.pNext = &renderingInfo;
    pipelineInfo.stageCount = static_cast<uint32_t>(stages.size());
    pipelineInfo.pStages = stages.data();
    pipelineInfo.layout = executor->layout;
    pipelineInfo.pRasterizationState = &rasterInfo;
    pipelineInfo.pMultisampleState = &msInfo;

    auto pipeRes =
        executor->_device.createGraphicsPipeline(nullptr, pipelineInfo);

    if (taskModule)
      executor->_device.destroyShaderModule(taskModule);
    if (meshModule)
      executor->_device.destroyShaderModule(meshModule);

    if (pipeRes.result != vk::Result::eSuccess)
      return pipeRes.result;
    executor->pipeline = pipeRes.value;

    return executor;
  }

  void record(uint32_t x, uint32_t y, uint32_t z, const TParams &pushConstants,
              StreamHandle handle = cDefaultStream) {
    Stream *stream = this->ctx->get_stream(handle);
    if (!stream)
      return;

    DirectParams params = {this, pushConstants, x, y, z};
    stream->record(_direct_record_cb, &params);
  }

  void record_indirect(const AllocationInfo &indirectBuffer,
                       const TParams &pushConstants,
                       StreamHandle handle = cDefaultStream) {
    Stream *stream = this->ctx->get_stream(handle);
    if (!stream)
      return;

    auto params = IndirectParams{
        .self = this, .pushConstants = pushConstants, .alloc = indirectBuffer};
    stream->record(_indirect_record_cb, &params);
  }

private:
  MeshExecutor() = default;

  struct DirectParams {
    MeshExecutor<TCode, TParams, TResources> *self;
    TParams const &pushConstants;
    uint32_t x, y, z;
  };

  struct IndirectParams {
    MeshExecutor<TCode, TParams, TResources> *self;
    TParams const &pushConstants;
    AllocationInfo const &alloc;
  };

  static void _direct_record_cb(vk::CommandBuffer cmd, void *data) {
    auto p = static_cast<DirectParams *>(data);

    vk::RenderingInfo renderingInfo{};
    renderingInfo.renderArea = vk::Rect2D{{0, 0}, {1, 1}};
    renderingInfo.layerCount = 1;

    cmd.beginRendering(renderingInfo);
    p->self->__bind_push_constants(cmd, vk::PipelineBindPoint::eGraphics,
                                   p->pushConstants);
    cmd.drawMeshTasksEXT(p->x, p->y, p->z);
    cmd.endRendering();
  }

  static void _indirect_record_cb(vk::CommandBuffer cmd, void *data) {
    auto p = static_cast<IndirectParams *>(data);

    vk::RenderingInfo renderingInfo{};
    renderingInfo.renderArea = vk::Rect2D{{0, 0}, {1, 1}};
    renderingInfo.layerCount = 1;

    cmd.beginRendering(renderingInfo);
    p->self->__bind_push_constants(cmd, vk::PipelineBindPoint::eGraphics,
                                   p->pushConstants);
    cmd.drawMeshTasksIndirectEXT(p->alloc.buffer, p->alloc.offset, 1, 0);
    cmd.endRendering();
  }
};

} // namespace pictor
