/* SPDX-License-Identifier: MPL-2.0 */

#pragma once

#include <vulkan/vulkan.hpp>

#include <array>
#include <cstdint>
#include <string>

namespace pictor {

struct DeviceInfo {
    uint32_t vendorID{0};
    uint32_t deviceID{0};
    uint32_t driverVersion{0};
    std::string deviceName;
    vk::PhysicalDeviceType deviceType{vk::PhysicalDeviceType::eOther};

    uint32_t maxComputeSharedMemorySize{0};
    uint32_t maxComputeWorkGroupInvocations{0};
    std::array<uint32_t, 3> maxComputeWorkGroupSize{0, 0, 0};
    /* Lets the sort replace its hardcoded kSortMaxGrid = 65535 with the device's real limit. */
    std::array<uint32_t, 3> maxComputeWorkGroupCount{0, 0, 0};

    vk::DeviceSize maxStorageBufferRange{0};
    vk::DeviceSize minStorageBufferOffsetAlignment{0};
    vk::DeviceSize nonCoherentAtomSize{0};

    float timestampPeriod{0.0f};
    uint32_t timestampValidBits{0};

    uint32_t subgroup_size{0};
    bool has_subgroup_partitioned{false};
};

} // namespace pictor
