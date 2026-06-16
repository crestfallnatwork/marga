#pragma once

#include "base_executor.hpp"
#include <iostream>

namespace pictor {

template <typename TCode, typename TParams, typename TResources = void>
class ComputeExecutor : public BaseExecutor<TParams, TResources> {
public:
  static Expected<std::unique_ptr<ComputeExecutor<TCode, TParams, TResources>>,
                  vk::Result>
  create(Context &ctx,
         const vk::SpecializationInfo *specInfo = nullptr) {

    auto compileResult = ctx.shader_compiler->compile(std::string(TCode::source),
                                          std::string(TCode::entry_point),
                                          ShaderStage::Compute);
    if (!compileResult.success) {
      return vk::Result::eErrorInitializationFailed;
    }

    return create_from_spirv(ctx, compileResult.spirv,
                             std::string(TCode::entry_point), specInfo);
  }

  static Expected<std::unique_ptr<ComputeExecutor<TCode, TParams, TResources>>,
                  vk::Result>
  create_from_spirv(Context &ctx, const std::vector<uint32_t> &computeShader,
                    const std::string &entry = "main",
                    const vk::SpecializationInfo *specInfo = nullptr) {

    auto executor = std::unique_ptr<ComputeExecutor<TCode, TParams, TResources>>(
        new ComputeExecutor<TCode, TParams, TResources>());
    executor->ctx = &ctx;

    // Auto-Binding: Get bindings from the Resources struct if it exists
    std::vector<vk::DescriptorSetLayoutBinding> bindings;
    if constexpr (!std::is_void_v<TResources>) {
      bindings = TResources::get_bindings();
    }

    auto layoutRes = executor->_create_layout(
         vk::ShaderStageFlagBits::eCompute, bindings);
    if (layoutRes != vk::Result::eSuccess)
      return layoutRes;

    auto moduleRes = executor->_create_module(computeShader);
    if (moduleRes.result != vk::Result::eSuccess)
      return moduleRes.result;
    vk::ShaderModule computeModule = moduleRes.value;

    vk::ComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.stage = {{},
                          vk::ShaderStageFlagBits::eCompute,
                          computeModule,
                          "main",
                          specInfo};
    pipelineInfo.layout = executor->layout;

    auto pipeRes =
        executor->_device.createComputePipeline(nullptr, pipelineInfo);

    executor->_device.destroyShaderModule(computeModule);

    if (pipeRes.result != vk::Result::eSuccess) {
      std::cerr << "Compute pipeline creation failed" << std::endl;
      return pipeRes.result;
    }
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

    IndirectParams params = {this, pushConstants, indirectBuffer.buffer,
                             indirectBuffer.offset};
    stream->record(_indirect_record_cb, &params);
  }

  private:
  ComputeExecutor() = default;

  struct DirectParams {
    ComputeExecutor<TCode, TParams, TResources> *self;
    TParams const &pushConstants;
    uint32_t x, y, z;
  };

  struct IndirectParams {
    ComputeExecutor<TCode, TParams, TResources> *self;
    TParams const &pushConstants;
    vk::Buffer buffer;
    size_t offset;
  };

  static void _direct_record_cb(vk::CommandBuffer cmd, void *data) {
    auto p = static_cast<DirectParams *>(data);
    p->self->__bind_push_constants(cmd, vk::PipelineBindPoint::eCompute,
                                   p->pushConstants);
    cmd.dispatch(p->x, p->y, p->z);
  }

  static void _indirect_record_cb(vk::CommandBuffer cmd, void *data) {
    auto p = static_cast<IndirectParams *>(data);
    p->self->__bind_push_constants(cmd, vk::PipelineBindPoint::eCompute,
                                   p->pushConstants);
    cmd.dispatchIndirect(p->buffer, p->offset);
  }
  };


} // namespace pictor
