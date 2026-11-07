/* SPDX-License-Identifier: MPL-2.0 */
/* Author: Himanshu Gupta <himanshu3@iisc.ac.in> */

#pragma once

#include "pictor_core/context/result_type.hpp"
#include "pictor_core/execution/shader_defines.hpp"
#include "pictor_core/execution/shader_sources.hpp"

#include <vulkan/vulkan.hpp>

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace pictor {

enum class ShaderStage {
    Compute,
    Task,
    Mesh,
    Fragment
};

struct CompileResult {
    std::vector<uint32_t> spirv;
    std::string diagnostics;
    std::array<uint32_t, 3> threadGroupSize{1, 1, 1};
    bool success{false};
};

class ShaderCompiler {
  public:
    static Expected<std::unique_ptr<ShaderCompiler>, vk::Result> create();
    ~ShaderCompiler();

    void set_debug_info(bool enabled)
    {
        _debug_info = enabled;
    }

    [[nodiscard]] bool debug_info() const
    {
        return _debug_info;
    }

    CompileResult compile(const std::string& source, const std::string& entryPoint,
                          ShaderStage stage, uint32_t subgroupSize = 32,
                          CompileOptions options = {});

    CompileResult compile(const ShaderSources& sources, const std::string& entryPoint,
                          ShaderStage stage, uint32_t subgroupSize = 32,
                          CompileOptions options = {});

  private:
    ShaderCompiler();

    struct Impl;
    std::unique_ptr<Impl> _impl;
    bool _debug_info{false};
    std::string _source_root;
};

} // namespace pictor
