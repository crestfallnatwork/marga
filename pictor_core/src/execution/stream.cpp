/* SPDX-License-Identifier: MPL-2.0 */
/* Author: Himanshu Gupta <himanshu3@iisc.ac.in> */

#include "pictor_core/execution/stream.hpp"

#include <algorithm>

namespace pictor {

Stream::Stream(vk::Device dev, uint32_t queue_family_index) : device(dev)
{
    vk::CommandPoolCreateInfo poolInfo{};
    poolInfo.queueFamilyIndex = queue_family_index;
    poolInfo.flags            = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;

    auto poolRes = device.createCommandPool(poolInfo);
    if (poolRes.result == vk::Result::eSuccess) {
        command_pool = poolRes.value;

        vk::CommandBufferAllocateInfo allocInfo{};
        allocInfo.commandPool        = command_pool;
        allocInfo.level              = vk::CommandBufferLevel::ePrimary;
        allocInfo.commandBufferCount = 2;

        auto cbRes = device.allocateCommandBuffers(allocInfo);
        if (cbRes.result == vk::Result::eSuccess) {
            _command_buffers[0] = cbRes.value[0];
            _command_buffers[1] = cbRes.value[1];
        }

        vk::FenceCreateInfo fenceInfo{};
        fenceInfo.flags = vk::FenceCreateFlagBits::eSignaled;
        _fences[0]      = device.createFence(fenceInfo).value;
        _fences[1]      = device.createFence(fenceInfo).value;
    }
}

Stream::~Stream()
{
    if (device) {
        synchronize();
        for (auto f : _fences)
            device.destroyFence(f);
        if (command_pool)
            device.destroyCommandPool(command_pool);
    }
}

vk::Result Stream::begin()
{
    if (_recording)
        return vk::Result::eSuccess;

    (void)device.waitForFences(1, &_fences[_current_frame], VK_TRUE, UINT64_MAX);
    (void)device.resetFences(1, &_fences[_current_frame]);

    (void)_command_buffers[_current_frame].reset();

    vk::CommandBufferBeginInfo beginInfo{};
    beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;

    vk::Result res = _command_buffers[_current_frame].begin(beginInfo);
    if (res == vk::Result::eSuccess) {
        _recording = true;
    }
    return res;
}

vk::Result Stream::end()
{
    if (!_recording)
        return vk::Result::eSuccess;

    vk::Result res = _command_buffers[_current_frame].end();
    if (res == vk::Result::eSuccess) {
        _recording = false;
    }
    return res;
}

vk::Result Stream::flush(vk::Queue queue)
{
    if (!_recording)
        return vk::Result::eSuccess;

    vk::Result res = end();
    if (res != vk::Result::eSuccess)
        return res;

    vk::SubmitInfo submitInfo{};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers    = &_command_buffers[_current_frame];

    res = queue.submit(1, &submitInfo, _fences[_current_frame]);
    if (res != vk::Result::eSuccess)
        return res;

    _current_frame = (_current_frame + 1) % 2;
    return vk::Result::eSuccess;
}

vk::Result Stream::fsync(vk::Queue queue)
{
    auto res = flush(queue);
    if (res != vk::Result::eSuccess)
        return res;
    synchronize();
    return vk::Result::eSuccess;
}

void Stream::synchronize()
{
    (void)device.waitForFences(2, _fences.data(), VK_TRUE, UINT64_MAX);
}

void Stream::barrier(const AllocationInfo& resource, Access from, Access to)
{
    if (!_recording) {
        if (begin() != vk::Result::eSuccess)
            return;
    }

    auto fromInfo = get_vulkan_access_info(from);
    auto toInfo   = get_vulkan_access_info(to);

    vk::BufferMemoryBarrier2 bufferBarrier{};
    bufferBarrier.srcStageMask  = fromInfo.stage;
    bufferBarrier.srcAccessMask = fromInfo.access;
    bufferBarrier.dstStageMask  = toInfo.stage;
    bufferBarrier.dstAccessMask = toInfo.access;
    bufferBarrier.buffer        = resource.buffer;
    bufferBarrier.offset        = resource.offset;
    bufferBarrier.size          = resource.size;

    vk::DependencyInfo dependencyInfo{};
    dependencyInfo.bufferMemoryBarrierCount = 1;
    dependencyInfo.pBufferMemoryBarriers    = &bufferBarrier;

    _command_buffers[_current_frame].pipelineBarrier2(dependencyInfo);
}

void Stream::barrier(const std::vector<ResourceBarrier>& barriers)
{
    if (!_recording) {
        if (begin() != vk::Result::eSuccess)
            return;
    }

    std::vector<vk::BufferMemoryBarrier2> bufferBarriers;
    bufferBarriers.reserve(barriers.size());

    for (const auto& b : barriers) {
        auto fromInfo = get_vulkan_access_info(b.from);
        auto toInfo   = get_vulkan_access_info(b.to);

        vk::BufferMemoryBarrier2 bufferBarrier{};
        bufferBarrier.srcStageMask  = fromInfo.stage;
        bufferBarrier.srcAccessMask = fromInfo.access;
        bufferBarrier.dstStageMask  = toInfo.stage;
        bufferBarrier.dstAccessMask = toInfo.access;
        bufferBarrier.buffer        = b.resource->buffer;
        bufferBarrier.offset        = b.resource->offset;
        bufferBarrier.size          = b.resource->size;

        bufferBarriers.push_back(bufferBarrier);
    }

    vk::DependencyInfo dependencyInfo{};
    dependencyInfo.bufferMemoryBarrierCount = static_cast<uint32_t>(bufferBarriers.size());
    dependencyInfo.pBufferMemoryBarriers    = bufferBarriers.data();

    _command_buffers[_current_frame].pipelineBarrier2(dependencyInfo);
}

void Stream::record(RecordCallback cb, void* user_data)
{
    if (!_recording) {
        if (begin() != vk::Result::eSuccess)
            return;
    }

    if (cb) {
        cb(_command_buffers[_current_frame], user_data);
    }
}

void Stream::_copy(const AllocationInfo& src, const AllocationInfo& dst)
{
    if (!_recording) {
        if (begin() != vk::Result::eSuccess)
            return;
    }

    vk::BufferCopy copyRegion{};
    copyRegion.srcOffset = src.offset;
    copyRegion.dstOffset = dst.offset;
    copyRegion.size      = std::min(src.size, dst.size);

    _command_buffers[_current_frame].copyBuffer(src.buffer, dst.buffer, copyRegion);
}

void Stream::_fill(const AllocationInfo& dst, size_t offset, size_t size, uint32_t value)
{
    if (!_recording) {
        if (begin() != vk::Result::eSuccess)
            return;
    }
    _command_buffers[_current_frame].fillBuffer(dst.buffer, dst.offset + offset, size, value);
}

void Stream::_update(const AllocationInfo& dst, size_t offset, size_t size, const void* data)
{
    if (!_recording) {
        if (begin() != vk::Result::eSuccess)
            return;
    }
    _command_buffers[_current_frame].updateBuffer(dst.buffer, dst.offset + offset, size, data);
}

} // namespace pictor
