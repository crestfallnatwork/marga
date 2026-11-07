/* SPDX-License-Identifier: MPL-2.0 */

#include "pictor_core/context/context.hpp"

#include <cstdint>
#include <cstdio>

using namespace pictor;

namespace {

bool g_ok = true;

template <typename A, typename B> void check_eq(const A& got, const B& want, const char* field)
{
    if (!(got == want)) {
        std::fprintf(stderr, "device info mismatch: %s\n", field);
        g_ok = false;
    }
}

void check(bool cond, const char* field)
{
    if (!cond) {
        std::fprintf(stderr, "device info sanity check failed: %s\n", field);
        g_ok = false;
    }
}

} // namespace

int main()
{
    auto ctx_res = Context::create();
    if (!ctx_res.ok()) {
        std::fprintf(stderr, "failed to create pictor context (vulkan %d)\n",
                     static_cast<int>(ctx_res.err()));
        return 1;
    }
    auto ctx = std::move(ctx_res.val());

    const DeviceInfo& info = ctx->device_info;
    const auto props       = ctx->physical_device.getProperties();
    const auto& limits     = props.limits;

    check_eq(info.vendorID, props.vendorID, "vendorID");
    check_eq(info.deviceID, props.deviceID, "deviceID");
    check_eq(info.driverVersion, props.driverVersion, "driverVersion");
    check_eq(info.deviceName, props.deviceName.data(), "deviceName");
    check_eq(info.deviceType, props.deviceType, "deviceType");

    check_eq(info.maxComputeSharedMemorySize, limits.maxComputeSharedMemorySize,
             "maxComputeSharedMemorySize");
    check_eq(info.maxComputeWorkGroupInvocations, limits.maxComputeWorkGroupInvocations,
             "maxComputeWorkGroupInvocations");
    for (uint32_t i = 0; i < 3; ++i) {
        check_eq(info.maxComputeWorkGroupSize[i], limits.maxComputeWorkGroupSize[i],
                 "maxComputeWorkGroupSize");
        check_eq(info.maxComputeWorkGroupCount[i], limits.maxComputeWorkGroupCount[i],
                 "maxComputeWorkGroupCount");
    }

    check_eq(info.maxStorageBufferRange, limits.maxStorageBufferRange, "maxStorageBufferRange");
    check_eq(info.minStorageBufferOffsetAlignment, limits.minStorageBufferOffsetAlignment,
             "minStorageBufferOffsetAlignment");
    check_eq(info.nonCoherentAtomSize, limits.nonCoherentAtomSize, "nonCoherentAtomSize");

    check_eq(info.timestampPeriod, limits.timestampPeriod, "timestampPeriod");
    check_eq(
        info.timestampValidBits,
        ctx->physical_device.getQueueFamilyProperties()[ctx->queue_family_index].timestampValidBits,
        "timestampValidBits");

    check_eq(info.subgroup_size, ctx->subgroup_size, "subgroup_size");

    check(info.maxComputeSharedMemorySize > 0, "maxComputeSharedMemorySize > 0");
    check(info.maxComputeWorkGroupInvocations > 0, "maxComputeWorkGroupInvocations > 0");
    for (uint32_t i = 0; i < 3; ++i) {
        check(info.maxComputeWorkGroupSize[i] > 0, "maxComputeWorkGroupSize > 0");
        check(info.maxComputeWorkGroupSize[i] <= info.maxComputeWorkGroupInvocations,
              "maxComputeWorkGroupSize <= maxComputeWorkGroupInvocations");
        check(info.maxComputeWorkGroupCount[i] > 0, "maxComputeWorkGroupCount > 0");
    }
    check(info.maxStorageBufferRange > 0, "maxStorageBufferRange > 0");
    check(info.minStorageBufferOffsetAlignment > 0, "minStorageBufferOffsetAlignment > 0");
    check(info.nonCoherentAtomSize > 0, "nonCoherentAtomSize > 0");
    check(info.subgroup_size > 0, "subgroup_size > 0");

    // Capability flags come from extension availability rather than getProperties(), so they are
    // checked against a direct enumeration instead.
    bool can_partition = false;
    auto available     = ctx->physical_device.enumerateDeviceExtensionProperties();
    if (available.result == vk::Result::eSuccess) {
        for (const auto& extension : available.value) {
            if (std::string(extension.extensionName.data()) ==
                VK_NV_SHADER_SUBGROUP_PARTITIONED_EXTENSION_NAME) {
                can_partition = true;
                break;
            }
        }
    }
    check_eq(info.has_subgroup_partitioned, can_partition, "has_subgroup_partitioned");

    std::printf("%s: %s (subgroup %u, shared %u B)\n",
                g_ok ? "device info test passed" : "DEVICE INFO TEST FAILED",
                info.deviceName.c_str(), info.subgroup_size, info.maxComputeSharedMemorySize);
    return g_ok ? 0 : 1;
}
