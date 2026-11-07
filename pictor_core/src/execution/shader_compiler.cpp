/* SPDX-License-Identifier: MPL-2.0 */
/* Author: Himanshu Gupta <himanshu3@iisc.ac.in> */

#include "pictor_core/execution/shader_compiler.hpp"

#include "pictor_core/execution/shader_sources.hpp"

#include <slang-com-ptr.h>
#include <slang.h>

#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <span>
#include <string>
#include <vector>

namespace pictor {

namespace {

std::string normalizePath(const std::string& path)
{
    bool absolute = !path.empty() && path.front() == '/';
    std::vector<std::string> parts;
    size_t start = 0;
    while (start <= path.size()) {
        size_t end = path.find('/', start);
        if (end == std::string::npos) {
            end = path.size();
        }
        std::string part = path.substr(start, end - start);
        if (part == "..") {
            if (!parts.empty()) {
                parts.pop_back();
            }
        } else if (!part.empty() && part != ".") {
            parts.push_back(std::move(part));
        }
        if (end == path.size()) {
            break;
        }
        start = end + 1;
    }
    std::string result = absolute ? "/" : "";
    for (size_t i = 0; i < parts.size(); ++i) {
        if (i > 0) {
            result += '/';
        }
        result += parts[i];
    }
    return result;
}

class MemoryFileSystem final : public ISlangFileSystemExt {
  public:
    MemoryFileSystem(std::span<const ShaderSourceFile> files, std::string sourceRoot)
        : _sourceRoot(std::move(sourceRoot))
    {
        while (_sourceRoot.size() > 1 && _sourceRoot.back() == '/') {
            _sourceRoot.pop_back();
        }
        for (const auto& file : files) {
            _files.emplace(std::string(file.path), std::string(file.code));
        }
    }

    SLANG_NO_THROW SlangResult SLANG_MCALL queryInterface(const SlangUUID& uuid,
                                                          void** outObject) SLANG_OVERRIDE
    {
        if (!outObject) {
            return SLANG_E_INVALID_ARG;
        }
        if (uuid == ISlangFileSystem::getTypeGuid() || uuid == ISlangFileSystemExt::getTypeGuid() ||
            uuid == ISlangUnknown::getTypeGuid()) {
            *outObject = static_cast<ISlangFileSystemExt*>(this);
            addRef();
            return SLANG_OK;
        }
        return SLANG_E_NO_INTERFACE;
    }

    SLANG_NO_THROW uint32_t SLANG_MCALL addRef() SLANG_OVERRIDE
    {
        return ++_refCount;
    }

    SLANG_NO_THROW uint32_t SLANG_MCALL release() SLANG_OVERRIDE
    {
        return --_refCount;
    }

    SLANG_NO_THROW void* SLANG_MCALL castAs(const SlangUUID& uuid) SLANG_OVERRIDE
    {
        if (uuid == ISlangFileSystem::getTypeGuid() || uuid == ISlangFileSystemExt::getTypeGuid()) {
            return static_cast<ISlangFileSystemExt*>(this);
        }
        return nullptr;
    }

    SLANG_NO_THROW SlangResult SLANG_MCALL loadFile(const char* path,
                                                    ISlangBlob** outBlob) SLANG_OVERRIDE
    {
        auto it = _files.find(resolvePath(path));
        if (it == _files.end()) {
            return SLANG_E_NOT_FOUND;
        }
        *outBlob = slang_createBlob(it->second.data(), it->second.size());
        return *outBlob ? SLANG_OK : SLANG_E_OUT_OF_MEMORY;
    }

    SLANG_NO_THROW SlangResult SLANG_MCALL
    getFileUniqueIdentity(const char* path, ISlangBlob** outUniqueIdentity) SLANG_OVERRIDE
    {
        if (!path) {
            return SLANG_E_NOT_FOUND;
        }
        if (!_files.contains(resolvePath(path))) {
            return SLANG_E_NOT_FOUND;
        }
        std::string identity = realPath(path);
        *outUniqueIdentity   = slang_createBlob(identity.data(), identity.size());
        return *outUniqueIdentity ? SLANG_OK : SLANG_E_OUT_OF_MEMORY;
    }

    SLANG_NO_THROW SlangResult SLANG_MCALL calcCombinedPath(SlangPathType fromPathType,
                                                            const char* fromPath, const char* path,
                                                            ISlangBlob** pathOut) SLANG_OVERRIDE
    {
        if (path && path[0] == '/') {
            std::string absolutePath = normalizePath(path);
            *pathOut                 = slang_createBlob(absolutePath.data(), absolutePath.size());
            return *pathOut ? SLANG_OK : SLANG_E_OUT_OF_MEMORY;
        }
        std::string combined = fromPath ? fromPath : "";
        if (fromPathType == SLANG_PATH_TYPE_FILE) {
            auto slash = combined.find_last_of('/');
            combined   = (slash == std::string::npos) ? std::string() : combined.substr(0, slash);
        }
        if (!combined.empty() && combined.back() != '/') {
            combined += '/';
        }
        combined += path ? path : "";
        combined = normalizePath(combined);
        *pathOut = slang_createBlob(combined.data(), combined.size());
        return *pathOut ? SLANG_OK : SLANG_E_OUT_OF_MEMORY;
    }

    SLANG_NO_THROW SlangResult SLANG_MCALL getPathType(const char* path,
                                                       SlangPathType* pathTypeOut) SLANG_OVERRIDE
    {
        std::string candidate = resolvePath(path);
        if (_files.contains(candidate)) {
            *pathTypeOut = SLANG_PATH_TYPE_FILE;
            return SLANG_OK;
        }
        std::string prefix = candidate;
        if (!prefix.empty() && prefix.back() != '/') {
            prefix += '/';
        }
        for (const auto& [name, code] : _files) {
            if (name.starts_with(prefix)) {
                *pathTypeOut = SLANG_PATH_TYPE_DIRECTORY;
                return SLANG_OK;
            }
        }
        return SLANG_E_NOT_FOUND;
    }

    SLANG_NO_THROW SlangResult SLANG_MCALL getPath(PathKind kind, const char* path,
                                                   ISlangBlob** outPath) SLANG_OVERRIDE
    {
        if (kind == PathKind::OperatingSystem) {
            return SLANG_E_NOT_IMPLEMENTED;
        }
        std::string result = (kind == PathKind::Display) ? realPath(path) : resolvePath(path);
        *outPath           = slang_createBlob(result.data(), result.size());
        return *outPath ? SLANG_OK : SLANG_E_OUT_OF_MEMORY;
    }

    SLANG_NO_THROW void SLANG_MCALL clearCache() SLANG_OVERRIDE {}

    SLANG_NO_THROW SlangResult SLANG_MCALL enumeratePathContents(
        const char* path, FileSystemContentsCallBack callback, void* userData) SLANG_OVERRIDE
    {
        return SLANG_E_NOT_IMPLEMENTED;
    }

    SLANG_NO_THROW OSPathKind SLANG_MCALL getOSPathKind() SLANG_OVERRIDE
    {
        return OSPathKind::None;
    }

  private:
    std::string resolvePath(const char* path) const
    {
        std::string normalized = normalizePath(path ? path : "");
        if (_files.contains(normalized)) {
            return normalized;
        }
        if (!_sourceRoot.empty() && normalized.size() > _sourceRoot.size() &&
            normalized.starts_with(_sourceRoot) && normalized[_sourceRoot.size()] == '/') {
            std::string stripped = normalized.substr(_sourceRoot.size());
            if (_files.contains(stripped)) {
                return stripped;
            }
        }
        return normalized;
    }

    std::string realPath(const char* path) const
    {
        std::string resolved = resolvePath(path);
        if (!_sourceRoot.empty() && _files.contains(resolved)) {
            return _sourceRoot + resolved;
        }
        return resolved;
    }

    std::map<std::string, std::string> _files;
    std::string _sourceRoot;
    uint32_t _refCount{0};
};

} // namespace

struct ShaderCompiler::Impl {
    Slang::ComPtr<slang::IGlobalSession> globalSession;
};

ShaderCompiler::ShaderCompiler() : _impl(std::make_unique<Impl>()) {}

ShaderCompiler::~ShaderCompiler() = default;

Expected<std::unique_ptr<ShaderCompiler>, vk::Result> ShaderCompiler::create()
{
    auto compiler = std::unique_ptr<ShaderCompiler>(new ShaderCompiler());

    if (const char* root = std::getenv("PICTOR_SOURCE_ROOT")) {
        compiler->_source_root = root;
    }

    if (SLANG_FAILED(slang::createGlobalSession(compiler->_impl->globalSession.writeRef()))) {
        return vk::Result::eErrorInitializationFailed;
    }

    return compiler;
}

CompileResult ShaderCompiler::compile(const std::string& source, const std::string& entryPoint,
                                      ShaderStage stage, uint32_t subgroupSize,
                                      CompileOptions options)
{
    const std::array<ShaderSourceFile, 1> files = {ShaderSourceFile{"/pictor/jit.slang", source}};
    return compile(ShaderSources{"/pictor/jit.slang", {}, files}, entryPoint, stage, subgroupSize,
                   options);
}

CompileResult ShaderCompiler::compile(const ShaderSources& sources, const std::string& entryPoint,
                                      ShaderStage stage, uint32_t subgroupSize,
                                      CompileOptions options)
{
    CompileResult result;

    for (const auto& define : options.defines) {
        if (define.name == "WAVE_SIZE") {
            result.success = false;
            result.diagnostics =
                "CompileOptions must not define WAVE_SIZE; the subgroup size is a device fact "
                "and the runtime injects WAVE_SIZE from the device";
            return result;
        }
    }

    if (!_source_root.empty()) {
        for (const auto& file : sources.files) {
            std::filesystem::path path =
                std::filesystem::path(_source_root) / std::string(file.path).substr(1);
            std::error_code error;
            std::filesystem::create_directories(path.parent_path(), error);
            std::ofstream stream(path, std::ios::binary);
            stream.write(file.code.data(), static_cast<std::streamsize>(file.code.size()));
        }
    }

    MemoryFileSystem fileSystem(sources.files, _source_root);

    std::vector<std::string> searchPathStrings;
    searchPathStrings.reserve(sources.searchPaths.size());
    for (const auto& path : sources.searchPaths) {
        searchPathStrings.emplace_back(path);
    }
    std::vector<const char*> searchPaths;
    searchPaths.reserve(searchPathStrings.size());
    for (const auto& path : searchPathStrings) {
        searchPaths.push_back(path.c_str());
    }

    slang::TargetDesc targetDesc{};
    targetDesc.format  = SLANG_SPIRV;
    targetDesc.profile = _impl->globalSession->findProfile("spirv_1_5");

    std::string waveSizeStr = std::to_string(subgroupSize);
    std::vector<std::string> defineNames;
    std::vector<std::string> defineValues;
    defineNames.reserve(options.defines.size());
    defineValues.reserve(options.defines.size());
    for (const auto& define : options.defines) {
        defineNames.emplace_back(define.name);
        defineValues.emplace_back(define.value);
    }

    std::vector<slang::PreprocessorMacroDesc> macros;
    macros.reserve(1 + options.defines.size());
    macros.push_back(slang::PreprocessorMacroDesc{"WAVE_SIZE", waveSizeStr.c_str()});
    for (size_t i = 0; i < defineNames.size(); ++i) {
        macros.push_back(
            slang::PreprocessorMacroDesc{defineNames[i].c_str(), defineValues[i].c_str()});
    }

    slang::SessionDesc sessionDesc{};
    sessionDesc.targets                = &targetDesc;
    sessionDesc.targetCount            = 1;
    sessionDesc.preprocessorMacros     = macros.data();
    sessionDesc.preprocessorMacroCount = static_cast<uint32_t>(macros.size());
    sessionDesc.fileSystem             = &fileSystem;
    sessionDesc.searchPaths            = searchPaths.data();
    sessionDesc.searchPathCount        = static_cast<SlangInt>(searchPaths.size());

    std::array<slang::CompilerOptionEntry, 1> compilerOptions{};
    if (_debug_info) {
        compilerOptions[0].name            = slang::CompilerOptionName::DebugInformation;
        compilerOptions[0].value.kind      = slang::CompilerOptionValueKind::Int;
        compilerOptions[0].value.intValue0 = static_cast<int32_t>(SLANG_DEBUG_INFO_LEVEL_STANDARD);
        sessionDesc.compilerOptionEntries  = compilerOptions.data();
        sessionDesc.compilerOptionEntryCount = static_cast<uint32_t>(compilerOptions.size());
    }

    Slang::ComPtr<slang::ISession> session;
    if (SLANG_FAILED(_impl->globalSession->createSession(sessionDesc, session.writeRef()))) {
        result.success     = false;
        result.diagnostics = "Failed to create Slang session";
        return result;
    }

    Slang::ComPtr<slang::IBlob> diagnosticBlob;
    std::string entryPath(sources.entryPath);
    slang::IModule* module = nullptr;
    if (_source_root.empty()) {
        module = session->loadModule(entryPath.c_str(), diagnosticBlob.writeRef());
    } else {
        const ShaderSourceFile* entryFile = nullptr;
        for (const auto& file : sources.files) {
            if (file.path == sources.entryPath) {
                entryFile = &file;
                break;
            }
        }
        if (!entryFile) {
            result.success     = false;
            result.diagnostics = "the entry shader is not part of the source closure";
            return result;
        }
        std::string moduleName = entryPath.substr(entryPath.find_last_of('/') + 1);
        if (auto dot = moduleName.find('.'); dot != std::string::npos) {
            moduleName.erase(dot);
        }
        std::string entryRealPath = _source_root + entryPath;
        std::string entrySource(entryFile->code);
        module =
            session->loadModuleFromSourceString(moduleName.c_str(), entryRealPath.c_str(),
                                                entrySource.c_str(), diagnosticBlob.writeRef());
    }

    if (diagnosticBlob) {
        result.diagnostics = static_cast<const char*>(diagnosticBlob->getBufferPointer());
    }

    if (!module) {
        result.success = false;
        return result;
    }

    Slang::ComPtr<slang::IEntryPoint> entryPointPtr;
    if (SLANG_FAILED(module->findEntryPointByName(entryPoint.c_str(), entryPointPtr.writeRef()))) {
        result.success = false;
        return result;
    }

    std::vector<slang::IComponentType*> components = {module, entryPointPtr.get()};
    Slang::ComPtr<slang::IComponentType> program;
    if (SLANG_FAILED(
            session->createCompositeComponentType(components.data(), (SlangInt)components.size(),
                                                  program.writeRef(), diagnosticBlob.writeRef()))) {
        if (diagnosticBlob)
            result.diagnostics += static_cast<const char*>(diagnosticBlob->getBufferPointer());
        result.success = false;
        return result;
    }

    auto* programLayout = program->getLayout();
    if (programLayout) {
        auto* entryLayout = programLayout->getEntryPointByIndex(0);
        if (entryLayout) {
            std::array<SlangUInt, 3> threadGroupSize{1, 1, 1};
            entryLayout->getComputeThreadGroupSize(3, threadGroupSize.data());
            result.threadGroupSize = {static_cast<uint32_t>(threadGroupSize[0]),
                                      static_cast<uint32_t>(threadGroupSize[1]),
                                      static_cast<uint32_t>(threadGroupSize[2])};
        }
    }

    Slang::ComPtr<slang::IComponentType> linkedProgram;
    if (SLANG_FAILED(program->linkWithOptions(linkedProgram.writeRef(), 0, nullptr,
                                              diagnosticBlob.writeRef()))) {
        if (diagnosticBlob)
            result.diagnostics += static_cast<const char*>(diagnosticBlob->getBufferPointer());
        result.success = false;
        return result;
    }

    Slang::ComPtr<slang::IBlob> spirvBlob;
    if (SLANG_FAILED(linkedProgram->getEntryPointCode(0, 0, spirvBlob.writeRef(),
                                                      diagnosticBlob.writeRef()))) {
        if (diagnosticBlob)
            result.diagnostics += static_cast<const char*>(diagnosticBlob->getBufferPointer());
        result.success = false;
        return result;
    }

    if (spirvBlob) {
        const auto* code = static_cast<const uint32_t*>(spirvBlob->getBufferPointer());
        size_t size      = spirvBlob->getBufferSize() / sizeof(uint32_t);
        result.spirv.assign(code, code + size);
        result.success = true;
    } else {
        result.success = false;
    }

    return result;
}

} // namespace pictor
