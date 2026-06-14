#include "pictor_core/execution/shader_compiler.hpp"

#include <slang-com-ptr.h>
#include <slang.h>

namespace pictor {

struct ShaderCompiler::Impl {
  Slang::ComPtr<slang::IGlobalSession> globalSession;
  Slang::ComPtr<slang::ISession> session;

  SlangStage get_slang_stage(ShaderStage stage) {
    switch (stage) {
    case ShaderStage::Compute:
      return SLANG_STAGE_COMPUTE;
    case ShaderStage::Task:
      return SLANG_STAGE_AMPLIFICATION;
    case ShaderStage::Mesh:
      return SLANG_STAGE_MESH;
    case ShaderStage::Fragment:
      return SLANG_STAGE_FRAGMENT;
    }
    return SLANG_STAGE_NONE;
  }
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

  slang::TargetDesc targetDesc{};
  targetDesc.format = SLANG_SPIRV;
  targetDesc.profile =
      compiler->_impl->globalSession->findProfile("spirv_1_6");

  slang::SessionDesc sessionDesc{};
  sessionDesc.targets = &targetDesc;
  sessionDesc.targetCount = 1;

  if (SLANG_FAILED(compiler->_impl->globalSession->createSession(
          sessionDesc, compiler->_impl->session.writeRef()))) {
    return vk::Result::eErrorInitializationFailed;
  }

  return compiler;
}

CompileResult ShaderCompiler::compile(const std::string &source,
                                      const std::string &entryPoint,
                                      ShaderStage stage) {
  CompileResult result;

  Slang::ComPtr<slang::IBlob> diagnosticBlob;
  slang::IModule *module = _impl->session->loadModuleFromSourceString(
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

  // Compose a program from the module and entry point
  std::vector<slang::IComponentType *> components = {module,
                                                     entryPointPtr.get()};
  Slang::ComPtr<slang::IComponentType> program;
  if (SLANG_FAILED(_impl->session->createCompositeComponentType(
          components.data(), (SlangInt)components.size(), program.writeRef(),
          diagnosticBlob.writeRef()))) {
    if (diagnosticBlob)
      result.diagnostics += (const char *)diagnosticBlob->getBufferPointer();
    result.success = false;
    return result;
  }

  // Link and get SPIR-V code
  Slang::ComPtr<slang::IBlob> spirvBlob;
  if (SLANG_FAILED(program->getEntryPointCode(0, 0, spirvBlob.writeRef(),
                                              diagnosticBlob.writeRef()))) {
    if (diagnosticBlob)
      result.diagnostics += (const char *)diagnosticBlob->getBufferPointer();
    result.success = false;
    return result;
  }

  const uint32_t *code = (const uint32_t *)spirvBlob->getBufferPointer();
  size_t size = spirvBlob->getBufferSize() / sizeof(uint32_t);
  result.spirv.assign(code, code + size);
  result.success = true;

  return result;
}

} // namespace pictor
