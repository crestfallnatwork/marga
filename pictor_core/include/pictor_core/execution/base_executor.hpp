/* SPDX-License-Identifier: MPL-2.0 */
/* Author: Himanshu Gupta <himanshu3@iisc.ac.in> */

#pragma once

#include "pictor_core/context/context.hpp"

#include <vulkan/vulkan.hpp>

#include <fstream>
#include <iostream>
#include <string>
#include <utility>

namespace pictor {

// A tuner adjusts a shader's declared tunables for a specific device. Generated shader traits
// expose `using Tunables = ...;` (see docs/shader-codegen.md) and a tuner fills that struct in:
//
//     void tuner(const DeviceInfo& device, SomeShader::Tunables& tunables);
//
// The tuner receives the device limits and therefore owns the device-limits contract: it must
// return a configuration that fits the device. The executor does not model the kernel's
// shared-memory or workgroup costs, so it cannot reject an over-limit configuration before the
// driver does. Cost models live next to the primitive, not here.
//
// The first conjunct is load-bearing rather than stylistic: without it GCC 14 substitutes the
// second requires-expression's parameter list even for a code type that has no `Tunables`, and
// hard-errors instead of reporting the constraint as unsatisfied. The tuner is declared as a
// parameter (never value-initialised) so that a deduced forwarding reference like `Lambda&` is
// accepted and the callable is not required to be default-constructible.
template <typename TCode, typename TTuner>
concept TunerFor = requires { typename TCode::Tunables; } &&
                   requires(const DeviceInfo& device, TCode::Tunables& tunables, TTuner&& tuner) {
                       tuner(device, tunables);
                   };

// Default-constructs the shader's declared tunables, lets the tuner adjust them for this device,
// and returns the resulting shader defines.
template <typename TCode, typename TTuner>
[[nodiscard]] ShaderDefines tuned_defines(const DeviceInfo& device, TTuner&& tuner)
{
    typename TCode::Tunables tunables{};
    std::forward<TTuner>(tuner)(device, tunables);
    return tunables.defines();
}

template <typename TParams, typename TResources = void> class BaseExecutor {
  public:
    using ParamsType    = TParams;
    using ResourcesType = TResources;

    vk::Pipeline pipeline;
    vk::PipelineLayout layout;
    vk::DescriptorSetLayout desc_layout;

    virtual ~BaseExecutor()
    {
        if (_device) {
            if (pipeline)
                _device.destroyPipeline(pipeline);
            if (layout)
                _device.destroyPipelineLayout(layout);
            if (desc_layout)
                _device.destroyDescriptorSetLayout(desc_layout);
        }
    }

  protected:
    BaseExecutor() = default;

    vk::Device _device;
    Context* ctx;
    std::string _entry_name;

    vk::Result _create_layout(vk::ShaderStageFlags pushConstantStages,
                              const std::vector<vk::DescriptorSetLayoutBinding>& bindings)
    {
        _device = ctx->device;

        vk::PushConstantRange pushRange{};
        pushRange.stageFlags = pushConstantStages;
        pushRange.offset     = 0;
        pushRange.size       = sizeof(TParams);

        // Build the layout (always use push descriptor bit for Resources)
        vk::DescriptorSetLayoutCreateInfo layoutInfoDesc{};
        if (!bindings.empty()) {
            layoutInfoDesc.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
            layoutInfoDesc.bindingCount = static_cast<uint32_t>(bindings.size());
            layoutInfoDesc.pBindings    = bindings.data();

            auto descRes = _device.createDescriptorSetLayout(layoutInfoDesc);
            if (descRes.result != vk::Result::eSuccess)
                return descRes.result;
            desc_layout = descRes.value;
        }

        vk::PipelineLayoutCreateInfo layoutInfo{};
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges    = &pushRange;
        layoutInfo.setLayoutCount         = bindings.empty() ? 0 : 1;
        layoutInfo.pSetLayouts            = bindings.empty() ? nullptr : &desc_layout;

        auto layoutRes = _device.createPipelineLayout(layoutInfo);
        if (layoutRes.result != vk::Result::eSuccess)
            return layoutRes.result;
        layout = layoutRes.value;

        return vk::Result::eSuccess;
    }

    template <typename R = TResources>
    void __bind(vk::CommandBuffer cmd, vk::PipelineBindPoint bindPoint,
                const TParams& pushConstants, const R& resources)
        requires(!std::is_void_v<R>)
    {
        cmd.bindPipeline(bindPoint, pipeline);

        // Auto-bind descriptors using the generated metadata in TResources
        std::vector<vk::WriteDescriptorSet> writes;
        std::vector<vk::DescriptorBufferInfo> bufferInfos;
        resources.fill_writes(writes, bufferInfos);

        if (!writes.empty()) {
            cmd.pushDescriptorSetKHR(bindPoint, layout, 0, static_cast<uint32_t>(writes.size()),
                                     writes.data());
        }

        vk::ShaderStageFlags stages;
        if (bindPoint == vk::PipelineBindPoint::eCompute) {
            stages = vk::ShaderStageFlagBits::eCompute;
        } else {
            stages = vk::ShaderStageFlagBits::eTaskEXT | vk::ShaderStageFlagBits::eMeshEXT;
        }

        cmd.pushConstants(layout, stages, 0, sizeof(TParams), &pushConstants);
    }

    /**
     * Variant for pure BDA shaders (no Resources).
     */
    void __bind_push_constants(vk::CommandBuffer cmd, vk::PipelineBindPoint bindPoint,
                               const TParams& pushConstants)
    {
        cmd.bindPipeline(bindPoint, pipeline);

        vk::ShaderStageFlags stages;
        if (bindPoint == vk::PipelineBindPoint::eCompute) {
            stages = vk::ShaderStageFlagBits::eCompute;
        } else {
            stages = vk::ShaderStageFlagBits::eTaskEXT | vk::ShaderStageFlagBits::eMeshEXT;
        }

        cmd.pushConstants(layout, stages, 0, sizeof(TParams), &pushConstants);
    }

    std::vector<uint32_t> _load_shader(const std::string& path)
    {
        std::ifstream file(path, std::ios::ate | std::ios::binary);
        if (!file.is_open())
            return {};

        size_t fileSize = static_cast<size_t>(file.tellg());
        std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));
        file.seekg(0);
        file.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(fileSize));
        file.close();
        return buffer;
    }

    vk::ResultValue<vk::ShaderModule> _create_module(const std::vector<uint32_t>& code)
    {
        vk::ShaderModuleCreateInfo createInfo{};
        createInfo.codeSize = code.size() * sizeof(uint32_t);
        createInfo.pCode    = code.data();
        return _device.createShaderModule(createInfo);
    }
};

} // namespace pictor
