/* SPDX-License-Identifier: MPL-2.0 */

#include "pictor_core/execution/shader_compiler.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

using namespace pictor;

namespace {

constexpr const char* kSource = R"(
[shader("compute")]
[numthreads(4, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
}
)";

bool contains(const std::vector<uint32_t>& spirv, std::string_view needle)
{
    const auto* bytes = reinterpret_cast<const char*>(spirv.data());
    const size_t size = spirv.size() * sizeof(uint32_t);
    for (size_t i = 0; i + needle.size() <= size; ++i) {
        if (std::memcmp(bytes + i, needle.data(), needle.size()) == 0) {
            return true;
        }
    }
    return false;
}

void setEnv(const char* name, const char* value)
{
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

bool testDebugInfo()
{
    auto compiler_res = ShaderCompiler::create();
    if (!compiler_res.ok()) {
        std::fprintf(stderr, "failed to create the shader compiler (vulkan %d)\n",
                     static_cast<int>(compiler_res.err()));
        return false;
    }
    auto compiler = std::move(compiler_res.val());

    compiler->set_debug_info(false);
    auto plain = compiler->compile(std::string(kSource), "main", ShaderStage::Compute, 32);
    compiler->set_debug_info(true);
    auto debug = compiler->compile(std::string(kSource), "main", ShaderStage::Compute, 32);

    if (!plain.success || !debug.success) {
        std::fprintf(stderr, "compilation failed: %s%s\n", plain.diagnostics.c_str(),
                     debug.diagnostics.c_str());
        return false;
    }

    const bool ok = contains(debug.spirv, "NonSemantic.Shader.DebugInfo.100") &&
                    !contains(plain.spirv, "NonSemantic.Shader.DebugInfo.100") &&
                    debug.spirv.size() > plain.spirv.size();
    std::printf("plain %zu words, debug %zu words: %s\n", plain.spirv.size(), debug.spirv.size(),
                ok ? "shader debug info test passed" : "SHADER DEBUG INFO TEST FAILED");
    return ok;
}

bool testSourceRoot()
{
    namespace fs        = std::filesystem;
    const fs::path root = fs::temp_directory_path() / "pictor-source-root-test";
    fs::remove_all(root);
    setEnv("PICTOR_SOURCE_ROOT", root.string().c_str());

    auto compiler_res = ShaderCompiler::create();
    setEnv("PICTOR_SOURCE_ROOT", "");
    if (!compiler_res.ok()) {
        std::fprintf(stderr, "failed to create the shader compiler with a source root\n");
        return false;
    }
    auto compiler = std::move(compiler_res.val());
    compiler->set_debug_info(true);

    auto result = compiler->compile(std::string(kSource), "main", ShaderStage::Compute, 32);
    if (!result.success) {
        std::fprintf(stderr, "source-root compilation failed: %s\n", result.diagnostics.c_str());
        return false;
    }

    const fs::path written = root / "pictor" / "jit.slang";
    const bool wrote_file  = fs::exists(written);
    const bool embedded    = contains(result.spirv, written.string());
    fs::remove_all(root);

    std::printf("source root: file %s, real path %s: %s\n", wrote_file ? "written" : "missing",
                embedded ? "embedded" : "not embedded",
                wrote_file && embedded ? "source root test passed" : "SOURCE ROOT TEST FAILED");
    return wrote_file && embedded;
}

} // namespace

int main()
{
    return (testDebugInfo() && testSourceRoot()) ? 0 : 1;
}
