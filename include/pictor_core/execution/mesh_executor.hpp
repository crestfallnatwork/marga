#pragma once

#include "pictor_core/context/context.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <variant>
#include <vector>
#include <vulkan/vulkan.hpp>

namespace pictor {

template <typename T> class MeshExecutor {
public:
  using PushConstantType = T;

  vk::Pipeline pipeline;
  vk::PipelineLayout layout;

  static std::variant<std::unique_ptr<MeshExecutor<T>>, vk::Result>
  create(Context &ctx, const std::string &taskShaderPath,
         const std::string &meshShaderPath) {

    auto executor = std::unique_ptr<MeshExecutor<T>>(new MeshExecutor<T>());
    executor->_device = ctx.device;

    auto taskShader = executor->_load_shader(taskShaderPath);
    auto meshShader = executor->_load_shader(meshShaderPath);

    if (taskShader.empty() || meshShader.empty()) {
      return vk::Result::eErrorInitializationFailed;
    }

    vk::PushConstantRange pushRange{};
    pushRange.stageFlags =
        vk::ShaderStageFlagBits::eTaskEXT | vk::ShaderStageFlagBits::eMeshEXT;
    pushRange.offset = 0;
    pushRange.size = sizeof(T);

    vk::PipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &pushRange;

    auto layoutRes = executor->_device.createPipelineLayout(layoutInfo);
    if (layoutRes.result != vk::Result::eSuccess)
      return layoutRes.result;
    executor->layout = layoutRes.value;

    auto taskModuleRes = executor->_create_module(taskShader);
    if (taskModuleRes.result != vk::Result::eSuccess)
      return taskModuleRes.result;
    vk::ShaderModule taskModule = taskModuleRes.value;

    auto meshModuleRes = executor->_create_module(meshShader);
    if (meshModuleRes.result != vk::Result::eSuccess)
      return meshModuleRes.result;
    vk::ShaderModule meshModule = meshModuleRes.value;

    std::vector<vk::PipelineShaderStageCreateInfo> stages;
    stages.push_back(
        {{}, vk::ShaderStageFlagBits::eTaskEXT, taskModule, "main"});
    stages.push_back(
        {{}, vk::ShaderStageFlagBits::eMeshEXT, meshModule, "main"});

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

  ~MeshExecutor() {
    if (_device) {
      if (pipeline)
        _device.destroyPipeline(pipeline);
      if (layout)
        _device.destroyPipelineLayout(layout);
    }
  }

  void record(uint32_t x, uint32_t y, uint32_t z, const T &pushConstants,
              Context &ctx, StreamHandle handle = cDefaultStream) {
    Stream *stream = ctx.get_stream(handle);
    if (!stream)
      return;

    DirectParams params = {this, pushConstants, x, y, z};
    stream->record(_direct_record_cb, &params);
  }

  void record_indirect(const AllocationInfo &indirectBuffer, uint32_t drawCount,
                       uint32_t stride, const T &pushConstants, Context &ctx,
                       StreamHandle handle = cDefaultStream) {
    Stream *stream = ctx.get_stream(handle);
    if (!stream)
      return;

    IndirectParams params = {
        this,      pushConstants, indirectBuffer.buffer, indirectBuffer.offset,
        drawCount, stride};
    stream->record(_indirect_record_cb, &params);
  }

private:
  MeshExecutor() = default;

  struct DirectParams {
    MeshExecutor<T> *self;
    T pushConstants;
    uint32_t x, y, z;
  };

  struct IndirectParams {
    MeshExecutor<T> *self;
    T pushConstants;
    vk::Buffer buffer;
    size_t offset;
    uint32_t drawCount;
    uint32_t stride;
  };

  static void _direct_record_cb(vk::CommandBuffer cmd, void *data) {
    auto p = static_cast<DirectParams *>(data);

    vk::RenderingInfo renderingInfo{};
    renderingInfo.renderArea = vk::Rect2D{{0, 0}, {1, 1}};
    renderingInfo.layerCount = 1;

    cmd.beginRendering(renderingInfo);

    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, p->self->pipeline);
    cmd.pushConstants(p->self->layout,
                      vk::ShaderStageFlagBits::eTaskEXT |
                          vk::ShaderStageFlagBits::eMeshEXT,
                      0, sizeof(T), &p->pushConstants);
    cmd.drawMeshTasksEXT(p->x, p->y, p->z);

    cmd.endRendering();
  }

  static void _indirect_record_cb(vk::CommandBuffer cmd, void *data) {
    auto p = static_cast<IndirectParams *>(data);

    vk::RenderingInfo renderingInfo{};
    renderingInfo.renderArea = vk::Rect2D{{0, 0}, {1, 1}};
    renderingInfo.layerCount = 1;

    cmd.beginRendering(renderingInfo);

    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, p->self->pipeline);
    cmd.pushConstants(p->self->layout,
                      vk::ShaderStageFlagBits::eTaskEXT |
                          vk::ShaderStageFlagBits::eMeshEXT,
                      0, sizeof(T), &p->pushConstants);
    cmd.drawMeshTasksIndirectEXT(p->buffer, p->offset, p->drawCount, p->stride);

    cmd.endRendering();
  }

  std::vector<uint32_t> _load_shader(const std::string &path) {
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file.is_open())
      return {};

    size_t fileSize = (size_t)file.tellg();
    std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));
    file.seekg(0);
    file.read((char *)buffer.data(), fileSize);
    file.close();
    return buffer;
  }

  vk::ResultValue<vk::ShaderModule>
  _create_module(const std::vector<uint32_t> &code) {
    vk::ShaderModuleCreateInfo createInfo{};
    createInfo.codeSize = code.size() * sizeof(uint32_t);
    createInfo.pCode = code.data();
    return _device.createShaderModule(createInfo);
  }

  vk::Device _device;
};

} // namespace pictor
