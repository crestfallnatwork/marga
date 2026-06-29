/* SPDX-License-Identifier: MPL-2.0 */
/* Author: Himanshu Gupta <himanshu3@iisc.ac.in> */

#pragma once

#include <vulkan/vulkan.hpp>

namespace pictor {

struct AllocationInfo;

enum class Access {
    Read,
    Write,
    IndirectRead,
    TransferRead,
    TransferWrite
};

struct AccessInfo {
    vk::PipelineStageFlags2 stage;
    vk::AccessFlags2 access;
};

inline AccessInfo get_vulkan_access_info(Access access)
{
    switch (access) {
        case Access::Read:
            return {vk::PipelineStageFlagBits2::eTaskShaderEXT |
                        vk::PipelineStageFlagBits2::eMeshShaderEXT |
                        vk::PipelineStageFlagBits2::eComputeShader,
                    vk::AccessFlagBits2::eShaderRead};
        case Access::Write:
            return {vk::PipelineStageFlagBits2::eTaskShaderEXT |
                        vk::PipelineStageFlagBits2::eMeshShaderEXT |
                        vk::PipelineStageFlagBits2::eComputeShader,
                    vk::AccessFlagBits2::eShaderWrite};
        case Access::IndirectRead:
            return {vk::PipelineStageFlagBits2::eDrawIndirect,
                    vk::AccessFlagBits2::eIndirectCommandRead};
        case Access::TransferRead:
            return {vk::PipelineStageFlagBits2::eAllTransfer, vk::AccessFlagBits2::eTransferRead};
        case Access::TransferWrite:
            return {vk::PipelineStageFlagBits2::eAllTransfer, vk::AccessFlagBits2::eTransferWrite};
    }
    return {vk::PipelineStageFlagBits2::eNone, vk::AccessFlagBits2::eNone};
}

struct ResourceBarrier {
    const AllocationInfo* resource;
    Access from;
    Access to;
};

} // namespace pictor
