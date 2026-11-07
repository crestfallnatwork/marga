/* SPDX-License-Identifier: MPL-2.0 */

#include "pictor_core/profiling/gputrace.hpp"

#ifdef PICTOR_NGFX_SDK

#include <vulkan/vulkan.h>

#include <NGFX_GPUTrace_Vulkan.h>

#include <chrono>
#include <cstdlib>
#include <dlfcn.h>
#include <string>
#include <thread>

namespace pictor::gputrace {

namespace {

bool g_initialized = false;
VkQueue g_queue    = VK_NULL_HANDLE;

constexpr int kAttachWaitSteps = 50;
constexpr auto kAttachWaitStep = std::chrono::milliseconds(100);
constexpr int kStartRetries    = 20;
constexpr auto kStartRetryStep = std::chrono::milliseconds(50);

const char* install_dir()
{
    if (const char* dir = std::getenv("PICTOR_NGFX_INSTALL")) {
        return dir;
    }
    return "/opt/nsight-graphics/latest";
}

bool sdk_requested()
{
    const char* env = std::getenv("PICTOR_NGFX_SDK");
    return env && env[0] != '0';
}

} // namespace

bool initialize(vk::Queue queue)
{
    if (g_initialized) {
        return true;
    }
    if (!sdk_requested()) {
        return false;
    }
    if (queue) {
        g_queue = static_cast<VkQueue>(queue);
    }

    // The Nsight Graphics SDK is header-only; its implementation ships in the tool's injection
    // library. Linux has no self-injection API, so the runtime loads it itself - after Vulkan
    // init, because loading it earlier crashes some drivers during ICD negotiation. RTLD_NOLOAD
    // reuses the copy the tool already injected. See docs/profiling/gputrace.md.
    if (!dlopen("libWarpViz.Injection.so", RTLD_NOW | RTLD_NOLOAD)) {
        std::string injection =
            std::string(install_dir()) + "/target/linux-desktop-nomad-x64/libWarpViz.Injection.so";
        if (!dlopen(injection.c_str(), RTLD_NOW | RTLD_GLOBAL)) {
            return false;
        }
    }

    NGFX_GPUTrace_InitializeActivity_Vulkan_Params params{};
    params.version = NGFX_GPUTrace_InitializeActivity_Vulkan_Params_VER;
    if (NGFX_GPUTrace_InitializeActivity_Vulkan(&params) != NGFX_Result_Success) {
        return false;
    }

    g_initialized = true;
    return true;
}

bool available()
{
    return g_initialized;
}

void start()
{
    if (!g_initialized) {
        if (!sdk_requested()) {
            return;
        }
        for (int i = 0; i < kAttachWaitSteps && !initialize(); ++i) {
            std::this_thread::sleep_for(kAttachWaitStep);
        }
    }
    if (!g_initialized) {
        return;
    }

    for (int i = 0; i < kStartRetries; ++i) {
        NGFX_GPUTrace_StartTrace_Vulkan_Params params{};
        params.version = NGFX_GPUTrace_StartTrace_Vulkan_Params_VER;
        if (NGFX_GPUTrace_StartTrace_Vulkan(&params) == NGFX_Result_Success) {
            return;
        }
        std::this_thread::sleep_for(kStartRetryStep);
    }
}

void stop()
{
    if (!g_initialized) {
        return;
    }
    NGFX_GPUTrace_StopTrace_Vulkan_Params params{};
    params.version            = NGFX_GPUTrace_StopTrace_Vulkan_Params_VER;
    params.flags              = NGFX_GPUTrace_StopTraceFlag_None;
    params.queue              = g_queue;
    params.outputResources    = nullptr;
    params.numOutputResources = 0;
    (void)NGFX_GPUTrace_StopTrace_Vulkan(&params);
}

} // namespace pictor::gputrace

#else

namespace pictor::gputrace {

bool initialize(vk::Queue)
{
    return false;
}

bool available()
{
    return false;
}

void start() {}

void stop() {}

} // namespace pictor::gputrace

#endif
