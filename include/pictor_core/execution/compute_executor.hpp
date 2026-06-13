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

template <typename T> class ComputeExecutor {
public:
  using PushConstantType = T;

  vk::Pipeline pipeline;
  vk::PipelineLayout layout;

  static std::variant<std::unique_ptr<ComputeExecutor<T>>, vk::Result>
  create(Context &ctx, const std::string &computeShaderPath) {

    auto executor =
        std::unique_ptr<ComputeExecutor<T>>(new ComputeExecutor<T>());
    executor->_device = ctx.device;

    auto computeShader = executor->_load_shader(computeShaderPath);
    if (computeShader.empty()) {
      return vk::Result::eErrorInitializationFailed;
    }

    vk::PushConstantRange pushRange{};
    pushRange.stageFlags = vk::ShaderStageFlagBits::eCompute;
    pushRange.offset = 0;
    pushRange.size = sizeof(T);

    vk::PipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &pushRange;

    auto layoutRes = executor->_device.createPipelineLayout(layoutInfo);
    if (layoutRes.result != vk::Result::eSuccess)
      return layoutRes.result;
    executor->layout = layoutRes.value;

    auto moduleRes = executor->_create_module(computeShader);
    if (moduleRes.result != vk::Result::eSuccess)
      return moduleRes.result;
    vk::ShaderModule computeModule = moduleRes.value;

    vk::ComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.stage = {
        {}, vk::ShaderStageFlagBits::eCompute, computeModule, "main"};
    pipelineInfo.layout = executor->layout;

    auto pipeRes =
        executor->_device.createComputePipeline(nullptr, pipelineInfo);

    executor->_device.destroyShaderModule(computeModule);

    if (pipeRes.result != vk::Result::eSuccess)
      return pipeRes.result;
    executor->pipeline = pipeRes.value;

    return executor;
  }

  ~ComputeExecutor() {
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

  void record_indirect(const AllocationInfo &indirectBuffer, size_t offset,
                       const T &pushConstants, Context &ctx,
                       StreamHandle handle = cDefaultStream) {
    Stream *stream = ctx.get_stream(handle);
    if (!stream)
      return;

    IndirectParams params = {this, pushConstants, indirectBuffer.buffer,
                             offset};
    stream->record(_indirect_record_cb, &params);
  }

private:
  ComputeExecutor() = default;

  struct DirectParams {
    ComputeExecutor<T> *self;
    T pushConstants;
    uint32_t x, y, z;
  };

  struct IndirectParams {
    ComputeExecutor<T> *self;
    T pushConstants;
    vk::Buffer buffer;
    size_t offset;
  };

  static void _direct_record_cb(vk::CommandBuffer cmd, void *data) {
    auto p = static_cast<DirectParams *>(data);
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, p->self->pipeline);
    cmd.pushConstants(p->self->layout, vk::ShaderStageFlagBits::eCompute, 0,
                      sizeof(T), &p->pushConstants);
    cmd.dispatch(p->x, p->y, p->z);
  }

  static void _indirect_record_cb(vk::CommandBuffer cmd, void *data) {
    auto p = static_cast<IndirectParams *>(data);
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, p->self->pipeline);
    cmd.pushConstants(p->self->layout, vk::ShaderStageFlagBits::eCompute, 0,
                      sizeof(T), &p->pushConstants);
    cmd.dispatchIndirect(p->buffer, p->offset);
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
