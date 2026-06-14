#pragma once

#include <memory>
#include <string>
#include <variant>
#include <vector>
#include <vulkan/vulkan.hpp>

namespace pictor {

enum class ShaderStage { Compute, Task, Mesh, Fragment };

struct CompileResult {
  std::vector<uint32_t> spirv;
  std::string diagnostics;
  bool success{false};
};

class ShaderCompiler {
public:
  static std::variant<std::unique_ptr<ShaderCompiler>, vk::Result> create();
  ~ShaderCompiler();

  CompileResult compile(const std::string &source, const std::string &entryPoint,
                        ShaderStage stage);

private:
  ShaderCompiler();

  struct Impl;
  std::unique_ptr<Impl> _impl;
};

} // namespace pictor
