/* SPDX-License-Identifier: MPL-2.0 */
/* Author: Himanshu Gupta <himanshu3@iisc.ac.in> */

#pragma once

#include "base_executor.hpp"

#include <array>

namespace pictor {

template <typename TCode, typename TParams, typename TResources = void>
class ComputeExecutor : public BaseExecutor<TParams, TResources> {
  public:
    const uint32_t wgX;
    const uint32_t wgY;
    const uint32_t wgZ;

    static Expected<std::unique_ptr<ComputeExecutor<TCode, TParams, TResources>>, vk::Result>
    create(Context& ctx, const vk::SpecializationInfo* specInfo = nullptr,
           CompileOptions options = {})
    {

        auto compileResult =
            ctx.shader_compiler->compile(TCode::sources, std::string(TCode::entry_point),
                                         ShaderStage::Compute, ctx.subgroup_size, options);
        if (!compileResult.success) {
            return vk::Result::eErrorInitializationFailed;
        }

        return create_from_spirv(ctx, compileResult.spirv, std::string(TCode::entry_point),
                                 specInfo, compileResult.threadGroupSize);
    }

    // Tuned creation: `tuner` adjusts the shader's declared tunables for this device before the
    // module is compiled. `TunerFor` and `tuned_defines` own the callable shape, the device-limits
    // contract and the shared step.
    template <typename TTuner>
        requires TunerFor<TCode, TTuner>
    static Expected<std::unique_ptr<ComputeExecutor<TCode, TParams, TResources>>, vk::Result>
    create(Context& ctx, TTuner&& tuner, const vk::SpecializationInfo* specInfo = nullptr)
    {
        auto defines = tuned_defines<TCode>(ctx.device_info, std::forward<TTuner>(tuner));
        return create(ctx, specInfo, CompileOptions{defines.view()});
    }

    static Expected<std::unique_ptr<ComputeExecutor<TCode, TParams, TResources>>, vk::Result>
    create_from_spirv(Context& ctx, const std::vector<uint32_t>& computeShader,
                      const std::string& entry                = "main",
                      const vk::SpecializationInfo* specInfo  = nullptr,
                      std::array<uint32_t, 3> threadGroupSize = {1, 1, 1})
    {

        auto executor = std::unique_ptr<ComputeExecutor<TCode, TParams, TResources>>(
            new ComputeExecutor<TCode, TParams, TResources>(threadGroupSize[0], threadGroupSize[1],
                                                            threadGroupSize[2]));
        executor->ctx = &ctx;

        // Auto-Binding: Get bindings from the Resources struct if it exists
        std::vector<vk::DescriptorSetLayoutBinding> bindings;
        if constexpr (!std::is_void_v<TResources>) {
            bindings = TResources::get_bindings();
        }

        auto layoutRes = executor->_create_layout(vk::ShaderStageFlagBits::eCompute, bindings);
        if (layoutRes != vk::Result::eSuccess)
            return layoutRes;

        auto moduleRes = executor->_create_module(computeShader);
        if (moduleRes.result != vk::Result::eSuccess)
            return moduleRes.result;
        vk::ShaderModule computeModule = moduleRes.value;

        vk::ComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.stage = {
            {}, vk::ShaderStageFlagBits::eCompute, computeModule, "main", specInfo};
        pipelineInfo.layout = executor->layout;

        auto pipeRes = executor->_device.createComputePipeline(nullptr, pipelineInfo);

        executor->_device.destroyShaderModule(computeModule);

        if (pipeRes.result != vk::Result::eSuccess)
            return pipeRes.result;
        executor->pipeline    = pipeRes.value;
        executor->_entry_name = entry;
        executor->ctx->set_name(executor->pipeline, entry);

        return executor;
    }

    void record(uint32_t x, uint32_t y, uint32_t z, const TParams& pushConstants,
                StreamHandle handle = cDefaultStream)
    {
        Stream* stream = this->ctx->get_stream(handle);
        if (!stream)
            return;

        uint32_t x_ = (x + wgX - 1) / wgX;
        uint32_t y_ = (y + wgY - 1) / wgY;
        uint32_t z_ = (z + wgZ - 1) / wgZ;

        DirectParams params = {this, pushConstants, x_, y_, z_};
        stream->record(this->_entry_name.c_str(), _direct_record_cb, &params);
    }

    void record_indirect(const AllocationInfo& indirectBuffer, const TParams& pushConstants,
                         StreamHandle handle = cDefaultStream)
    {
        Stream* stream = this->ctx->get_stream(handle);
        if (!stream)
            return;

        IndirectParams params = {this, pushConstants, indirectBuffer.buffer, indirectBuffer.offset};
        stream->record(this->_entry_name.c_str(), _indirect_record_cb, &params);
    }

  private:
    ComputeExecutor() = default;
    ComputeExecutor(uint32_t wgX, uint32_t wgY, uint32_t wgZ) : wgX(wgX), wgY(wgY), wgZ(wgZ) {};

    struct DirectParams {
        ComputeExecutor<TCode, TParams, TResources>* self;
        TParams const& pushConstants;
        uint32_t x, y, z;
    };

    struct IndirectParams {
        ComputeExecutor<TCode, TParams, TResources>* self;
        TParams const& pushConstants;
        vk::Buffer buffer;
        size_t offset;
    };

    static void _direct_record_cb(vk::CommandBuffer cmd, void* data)
    {
        auto p = static_cast<DirectParams*>(data);
        p->self->__bind_push_constants(cmd, vk::PipelineBindPoint::eCompute, p->pushConstants);
        cmd.dispatch(p->x, p->y, p->z);
    }

    static void _indirect_record_cb(vk::CommandBuffer cmd, void* data)
    {
        auto p = static_cast<IndirectParams*>(data);
        p->self->__bind_push_constants(cmd, vk::PipelineBindPoint::eCompute, p->pushConstants);
        cmd.dispatchIndirect(p->buffer, p->offset);
    }
};

} // namespace pictor
