#pragma once

#include "pictor_core/context/result_type.hpp"

#include <vulkan/vulkan.hpp>

#include <memory>
#include <string>
#include <vector>

namespace pictor {

enum class ShaderStage { Compute, Task, Mesh, Fragment };

struct CompileResult {
  std::vector<uint32_t> spirv;
  std::string diagnostics;
  bool success{false};
};

class ShaderCompiler {
public:
  static Expected<std::unique_ptr<ShaderCompiler>, vk::Result> create();
  ~ShaderCompiler();

  CompileResult compile(const std::string &source,
                        const std::string &entryPoint, ShaderStage stage);

private:
  ShaderCompiler();

  struct Impl;
  std::unique_ptr<Impl> _impl;
};

} // namespace pictor
