#include "pictor_core/execution/shader_compiler.hpp"

#include <slang-com-ptr.h>
#include <slang.h>
#include <atomic>
#include <iostream>
#include <vector>

namespace pictor {

struct ShaderCompiler::Impl {
  Slang::ComPtr<slang::IGlobalSession> globalSession;
};

ShaderCompiler::ShaderCompiler() : _impl(std::make_unique<Impl>()) {}

ShaderCompiler::~ShaderCompiler() {}

Expected<std::unique_ptr<ShaderCompiler>, vk::Result>
ShaderCompiler::create() {
  auto compiler = std::unique_ptr<ShaderCompiler>(new ShaderCompiler());

  if (SLANG_FAILED(slang::createGlobalSession(
          compiler->_impl->globalSession.writeRef()))) {
    return vk::Result::eErrorInitializationFailed;
  }

  return compiler;
}

CompileResult ShaderCompiler::compile(const std::string &source,
                                      const std::string &entryPoint,
                                      ShaderStage stage,
                                      uint32_t subgroupSize) {
  CompileResult result;

  // Create a fresh session for each compilation to avoid internal dictionary collisions
  slang::TargetDesc targetDesc{};
  targetDesc.format = SLANG_SPIRV;
  targetDesc.profile = _impl->globalSession->findProfile("spirv_1_5");

  slang::SessionDesc sessionDesc{};
  sessionDesc.targets = &targetDesc;
  sessionDesc.targetCount = 1;

  std::string waveSizeStr = std::to_string(subgroupSize);
  slang::PreprocessorMacroDesc macros[] = {
      {"WAVE_SIZE", waveSizeStr.c_str()}
  };
  sessionDesc.preprocessorMacros = macros;
  sessionDesc.preprocessorMacroCount = 1;

  Slang::ComPtr<slang::ISession> session;
  if (SLANG_FAILED(_impl->globalSession->createSession(sessionDesc, session.writeRef()))) {
      result.success = false;
      result.diagnostics = "Failed to create Slang session";
      return result;
  }

  Slang::ComPtr<slang::IBlob> diagnosticBlob;
  slang::IModule *module = session->loadModuleFromSourceString(
      "pictor_jit", "jit.slang", source.c_str(), diagnosticBlob.writeRef());

  if (diagnosticBlob) {
    result.diagnostics = (const char *)diagnosticBlob->getBufferPointer();
  }

  if (!module) {
    result.success = false;
    return result;
  }

  Slang::ComPtr<slang::IEntryPoint> entryPointPtr;
  if (SLANG_FAILED(module->findEntryPointByName(entryPoint.c_str(),
                                                entryPointPtr.writeRef()))) {
    result.success = false;
    return result;
  }

  std::vector<slang::IComponentType *> components = {module,
                                                     entryPointPtr.get()};
  Slang::ComPtr<slang::IComponentType> program;
  if (SLANG_FAILED(session->createCompositeComponentType(
          components.data(), (SlangInt)components.size(), program.writeRef(),
          diagnosticBlob.writeRef()))) {
    if (diagnosticBlob)
      result.diagnostics += (const char *)diagnosticBlob->getBufferPointer();
    result.success = false;
    return result;
  }

  Slang::ComPtr<slang::IBlob> spirvBlob;
  if (SLANG_FAILED(program->getEntryPointCode(0, 0, spirvBlob.writeRef(),
                                              diagnosticBlob.writeRef()))) {
    if (diagnosticBlob)
      result.diagnostics += (const char *)diagnosticBlob->getBufferPointer();
    result.success = false;
    return result;
  }

  if (spirvBlob) {
    const uint32_t *code = (const uint32_t *)spirvBlob->getBufferPointer();
    size_t size = spirvBlob->getBufferSize() / sizeof(uint32_t);
    result.spirv.assign(code, code + size);
    result.success = true;
  } else {
    result.success = false;
  }

  return result;
}

} // namespace pictor
