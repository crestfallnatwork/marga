/* SPDX-License-Identifier: MPL-2.0 */
/* Author: Himanshu Gupta <himanshu3@iisc.ac.in> */

#define VMA_IMPLEMENTATION
#include <volk.h>

#include "pictor_core/context/context.hpp"

#include "pictor_core/profiling/gputrace.hpp"

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <limits>

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

namespace pictor {

Expected<std::unique_ptr<Context>, vk::Result> Context::create()
{
    if (volkInitialize() != VK_SUCCESS) {
        return vk::Result::eErrorInitializationFailed;
    }
    VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);

    auto ctx = std::unique_ptr<Context>(new Context());
#if defined(PICTOR_DEBUG_UTILS) && PICTOR_DEBUG_UTILS
    ctx->_debug_utils_enabled = true;
#endif
    if (const char* env = std::getenv("PICTOR_DEBUG_UTILS")) {
        ctx->_debug_utils_enabled = env[0] != '0';
    }
#if defined(PICTOR_SHADER_DEBUG_INFO) && PICTOR_SHADER_DEBUG_INFO
    ctx->_shader_debug_info = true;
#endif
    if (const char* env = std::getenv("PICTOR_SHADER_DEBUG_INFO")) {
        ctx->_shader_debug_info = env[0] != '0';
    }
    vk::Result res = ctx->_init();
    if (res == vk::Result::eSuccess) {
        (void)gputrace::initialize(ctx->queue);
        return ctx;
    }
    return res;
}

Context::~Context()
{
    wait_idle();
    _streams.clear();

    set_device_allocator(std::unique_ptr<Allocator>{nullptr});
    set_host_allocator(std::unique_ptr<Allocator>{nullptr});

    root_pinned_allocator.reset();
    root_pool_allocator.reset();

    if (device) {
        vma_allocator.destroy();
        device.destroy();
    }
    if (instance) {
        instance.destroy();
    }
}

StreamHandle Context::create_stream()
{
    auto handle = static_cast<StreamHandle>(_streams.size());
    _streams.emplace_back(std::make_unique<Stream>(device, queue_family_index));
    Stream* stream               = _streams.back().get();
    stream->_debug_utils_enabled = _debug_utils_enabled;
    stream->_set_debug_name(handle);
    return handle;
}

vk::Result Context::submit(StreamHandle handle)
{
    Stream* s = get_stream(handle);
    if (s) {
        return s->flush(queue);
    }
    return vk::Result::eErrorInvalidExternalHandle;
}
vk::Result Context::fsync(StreamHandle handle)
{
    Stream* s = get_stream(handle);
    if (s) {
        return s->fsync(queue);
    }
    return vk::Result::eErrorInvalidExternalHandle;
}

Stream* Context::get_stream(StreamHandle handle)
{
    if (handle < _streams.size()) {
        return _streams[handle].get();
    }
    return nullptr;
}

void Context::sync(StreamHandle handle)
{
    Stream* s = get_stream(handle);
    if (s) {
        s->synchronize();
    }
}

void Context::barrier(const AllocationInfo& resource, Access from, Access to, StreamHandle handle)
{
    Stream* s = get_stream(handle);
    if (s) {
        s->barrier(resource, from, to);
    }
}

void Context::barrier(const std::vector<ResourceBarrier>& barriers, StreamHandle handle)
{
    Stream* s = get_stream(handle);
    if (s) {
        s->barrier(barriers);
    }
}

void Context::copy(const AllocationInfo& src, const AllocationInfo& dst, StreamHandle stream)
{
    Stream* s = get_stream(stream);
    if (s) {
        s->_copy(src, dst);
    }
}

void Context::fill(const AllocationInfo& dst, uint32_t value, size_t offset, size_t size,
                   StreamHandle stream)
{
    if (size == std::numeric_limits<size_t>::max())
        size = dst.size;
    fill(dst, offset, size, value, stream);
}

void Context::fill(const AllocationInfo& dst, size_t offset, size_t size, uint32_t value,
                   StreamHandle stream)
{
    Stream* s = get_stream(stream);
    if (s) {
        s->_fill(dst, offset, size, value);
    }
}

void Context::update(const AllocationInfo& dst, size_t offset, size_t size, const void* data,
                     StreamHandle stream)
{
    Stream* s = get_stream(stream);
    if (s) {
        s->_update(dst, offset, size, data);
    }
}

void Context::deallocate(AllocationInfo& alloc)
{
    if (alloc.alloc)
        alloc.alloc->deallocate(alloc);
}
void Context::invalidate(AllocationInfo& alloc)
{
    alloc.alloc->invalidate(alloc);
}
void Context::flush(AllocationInfo& alloc)
{
    alloc.alloc->flush(alloc);
}

size_t Context::get_device_free_memory() const
{
    const auto* mem_props = vma_allocator.getMemoryProperties();
    auto budgets          = vma_allocator.getHeapBudgets();
    size_t usage          = 0;
    size_t budget         = 0;
    for (size_t i = 0; i < budgets.size(); i++) {
        if (!(mem_props->memoryHeaps[i].flags & vk::MemoryHeapFlagBits::eDeviceLocal)) {
            continue;
        }
        usage += budgets[i].usage;
        budget += budgets[i].budget;
    }
    return (budget - usage);
}

size_t Context::get_host_free_memory() const
{
    const auto* mem_props = vma_allocator.getMemoryProperties();
    auto budgets          = vma_allocator.getHeapBudgets();
    size_t usage          = 0;
    size_t budget         = 0;
    for (size_t i = 0; i < budgets.size(); i++) {
        if ((mem_props->memoryHeaps[i].flags & vk::MemoryHeapFlagBits::eDeviceLocal)) {
            continue;
        }
        usage += budgets[i].usage;
        budget += budgets[i].budget;
    }
    /* UMA fallback */
    if (budget == 0) {
        uint32_t uma_heap = std::numeric_limits<uint32_t>::max();
        for (uint32_t i = 0; i < mem_props->memoryTypeCount; i++) {
            if (!(mem_props->memoryTypes[i].propertyFlags &
                  vk::MemoryPropertyFlagBits::eHostVisible)) {
                continue;
            }
            uma_heap = mem_props->memoryTypes[i].heapIndex;
            break;
        }

        assert(uma_heap < budgets.size());
        usage  = budgets[uma_heap].usage;
        budget = budgets[uma_heap].budget;
    }
    return budget > usage ? (budget - usage) : 0;
}

void Context::set_host_allocator(std::unique_ptr<Allocator>&& allocator)
{
    if (pinned_allocator != root_pinned_allocator.get()) {
        delete pinned_allocator;
    }
    pinned_allocator = allocator.release();
}
void Context::set_device_allocator(std::unique_ptr<Allocator>&& allocator)
{
    if (pool_allocator != root_pool_allocator.get()) {
        delete pool_allocator;
    }
    pool_allocator = allocator.release();
}

vk::Result Context::_init()
{
    vk::Result res = _create_instance();
    if (res != vk::Result::eSuccess)
        return res;

    res = _select_physical_device();
    if (res != vk::Result::eSuccess)
        return res;

    res = _gather_device_info();
    if (res != vk::Result::eSuccess)
        return res;

    res = _create_logical_device();
    if (res != vk::Result::eSuccess)
        return res;

    res = _setup_vma();
    if (res != vk::Result::eSuccess)
        return res;

    res = _setup_allocators();
    if (res != vk::Result::eSuccess)
        return res;

    res = _setup_compiler();
    if (res != vk::Result::eSuccess)
        return res;

    res = _setup_streams();
    return res;
}

vk::Result Context::_create_instance()
{
    vk::ApplicationInfo appInfo{};
    appInfo.pApplicationName   = "pictor-core";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName        = "pictor-core";
    appInfo.engineVersion      = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion         = VK_API_VERSION_1_3;

    std::vector<const char*> layers = {};
#ifdef PICTOR_DEBUG
    layers.push_back("VK_LAYER_KHRONOS_validation");
#endif

    std::vector<const char*> extensions;
    if (_debug_utils_enabled) {
        bool available    = false;
        auto extensionRes = vk::enumerateInstanceExtensionProperties();
        if (extensionRes.result == vk::Result::eSuccess) {
            for (const auto& extension : extensionRes.value) {
                if (std::strcmp(extension.extensionName.data(),
                                VK_EXT_DEBUG_UTILS_EXTENSION_NAME) == 0) {
                    available = true;
                    break;
                }
            }
        }
        if (available) {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        } else {
            _debug_utils_enabled = false;
        }
    }

    vk::InstanceCreateInfo createInfo{};
    createInfo.pApplicationInfo        = &appInfo;
    createInfo.enabledLayerCount       = static_cast<uint32_t>(layers.size());
    createInfo.ppEnabledLayerNames     = layers.data();
    createInfo.enabledExtensionCount   = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

    auto res = vk::createInstance(createInfo);
    if (res.result == vk::Result::eSuccess) {
        instance = res.value;
        volkLoadInstance(instance);
        VULKAN_HPP_DEFAULT_DISPATCHER.init(instance);
    }
    return res.result;
}

vk::Result Context::_select_physical_device()
{
    auto devicesRes = instance.enumeratePhysicalDevices();
    if (devicesRes.result != vk::Result::eSuccess)
        return devicesRes.result;
    if (devicesRes.value.empty())
        return vk::Result::eErrorInitializationFailed;

    auto score_device = [&](vk::PhysicalDevice dev) {
        int score  = 0;
        auto props = dev.getProperties();

        if (props.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
            score += 1000;
        } else if (props.deviceType == vk::PhysicalDeviceType::eIntegratedGpu) {
            score += 500;
        }

        bool has_mesh_shader = false;
        auto extensionsRes   = dev.enumerateDeviceExtensionProperties();
        if (extensionsRes.result == vk::Result::eSuccess) {
            for (const auto& ext : extensionsRes.value) {
                if (std::string(ext.extensionName.data()) == VK_EXT_MESH_SHADER_EXTENSION_NAME) {
                    has_mesh_shader = true;
                    break;
                }
            }
        }

        if (!has_mesh_shader)
            return -1;

        vk::PhysicalDeviceMeshShaderFeaturesEXT meshFeatures;
        vk::PhysicalDeviceFeatures2 features2;
        features2.pNext = &meshFeatures;
        dev.getFeatures2(&features2);

        if (!meshFeatures.meshShader || !meshFeatures.taskShader)
            return -1;
        if (!dev.getFeatures().shaderInt64)
            return -1;

        return score;
    };

    vk::PhysicalDevice best_device = nullptr;
    int best_score                 = -1;

    for (const auto& dev : devicesRes.value) {
        int score = score_device(dev);
        if (score > best_score) {
            best_score  = score;
            best_device = dev;
        }
    }

    if (!best_device)
        return vk::Result::eErrorFeatureNotPresent;

    physical_device = best_device;
    return vk::Result::eSuccess;
}

vk::Result Context::_gather_device_info()
{
    vk::PhysicalDeviceSubgroupProperties subgroupProps;
    vk::PhysicalDeviceProperties2 props2;
    props2.pNext = &subgroupProps;
    physical_device.getProperties2(&props2);

    const auto& props  = props2.properties;
    const auto& limits = props.limits;

    subgroup_size = subgroupProps.subgroupSize;

    device_info.vendorID      = props.vendorID;
    device_info.deviceID      = props.deviceID;
    device_info.driverVersion = props.driverVersion;
    device_info.deviceName    = props.deviceName.data();
    device_info.deviceType    = props.deviceType;

    device_info.maxComputeSharedMemorySize     = limits.maxComputeSharedMemorySize;
    device_info.maxComputeWorkGroupInvocations = limits.maxComputeWorkGroupInvocations;
    device_info.maxComputeWorkGroupSize        = {limits.maxComputeWorkGroupSize[0],
                                                  limits.maxComputeWorkGroupSize[1],
                                                  limits.maxComputeWorkGroupSize[2]};
    device_info.maxComputeWorkGroupCount       = {limits.maxComputeWorkGroupCount[0],
                                                  limits.maxComputeWorkGroupCount[1],
                                                  limits.maxComputeWorkGroupCount[2]};

    device_info.maxStorageBufferRange           = limits.maxStorageBufferRange;
    device_info.minStorageBufferOffsetAlignment = limits.minStorageBufferOffsetAlignment;
    device_info.nonCoherentAtomSize             = limits.nonCoherentAtomSize;

    device_info.timestampPeriod = limits.timestampPeriod;

    device_info.subgroup_size = subgroup_size;

    // Capability flags that gate optional shader features. They live here with the rest of the
    // device facts so consumers never have to re-query the physical device.
    auto availableExtensions = physical_device.enumerateDeviceExtensionProperties();
    if (availableExtensions.result == vk::Result::eSuccess) {
        for (const auto& extension : availableExtensions.value) {
            if (std::string(extension.extensionName.data()) ==
                VK_NV_SHADER_SUBGROUP_PARTITIONED_EXTENSION_NAME) {
                device_info.has_subgroup_partitioned = true;
                break;
            }
        }
    }

    return vk::Result::eSuccess;
}

vk::Result Context::_create_logical_device()
{
    auto queueFamilies     = physical_device.getQueueFamilyProperties();
    uint32_t graphicsIndex = std::numeric_limits<uint32_t>::max();

    for (uint32_t i = 0; i < queueFamilies.size(); i++) {
        if ((queueFamilies[i].queueFlags & vk::QueueFlagBits::eGraphics) &&
            (queueFamilies[i].queueFlags & vk::QueueFlagBits::eCompute)) {
            graphicsIndex = i;
            break;
        }
    }

    if (graphicsIndex == std::numeric_limits<uint32_t>::max())
        return vk::Result::eErrorInitializationFailed;

    queue_family_index             = graphicsIndex;
    device_info.timestampValidBits = queueFamilies[graphicsIndex].timestampValidBits;

    float queuePriority = 1.0f;
    vk::DeviceQueueCreateInfo queueCreateInfo{};
    queueCreateInfo.queueFamilyIndex = queue_family_index;
    queueCreateInfo.queueCount       = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    std::vector<const char*> deviceExtensions = {
        VK_EXT_MESH_SHADER_EXTENSION_NAME,     VK_KHR_MAINTENANCE_4_EXTENSION_NAME,
        VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME, VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME,
        VK_EXT_MEMORY_BUDGET_EXTENSION_NAME,   VK_KHR_SHADER_ATOMIC_INT64_EXTENSION_NAME};

    // Required by any shader that partitions the subgroup (WaveMatch); enabled here so the
    // capability flag and the enabled-extension set can never disagree.
    if (device_info.has_subgroup_partitioned)
        deviceExtensions.push_back(VK_NV_SHADER_SUBGROUP_PARTITIONED_EXTENSION_NAME);

    vk::PhysicalDeviceShaderAtomicInt64Features atomicInt64Features{};
    atomicInt64Features.shaderBufferInt64Atomics = VK_TRUE;

    vk::PhysicalDeviceShaderSubgroupExtendedTypesFeatures extTypes = {};
    extTypes.shaderSubgroupExtendedTypes                           = VK_TRUE;
    extTypes.pNext                                                 = &atomicInt64Features;

    vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT atomicFloatFeatures;
    atomicFloatFeatures.shaderBufferFloat32AtomicAdd = VK_TRUE;
    atomicFloatFeatures.pNext                        = &extTypes;

    vk::PhysicalDeviceMeshShaderFeaturesEXT meshFeatures;
    meshFeatures.meshShader = VK_TRUE;
    meshFeatures.taskShader = VK_TRUE;
    meshFeatures.pNext      = &atomicFloatFeatures;

    vk::PhysicalDeviceBufferDeviceAddressFeatures bdaFeatures;
    bdaFeatures.bufferDeviceAddress = VK_TRUE;
    bdaFeatures.pNext               = &meshFeatures;

    vk::PhysicalDeviceSynchronization2Features sync2Features;
    sync2Features.synchronization2 = VK_TRUE;
    sync2Features.pNext            = &bdaFeatures;

    vk::PhysicalDeviceMaintenance4Features maintenance4;
    maintenance4.maintenance4 = VK_TRUE;
    maintenance4.pNext        = &sync2Features;

    vk::PhysicalDeviceDynamicRenderingFeatures dynamicRendering;
    dynamicRendering.dynamicRendering = VK_TRUE;
    dynamicRendering.pNext            = &maintenance4;

    vk::PhysicalDeviceFeatures2 deviceFeatures2;
    deviceFeatures2.features.shaderInt64 = VK_TRUE;
    deviceFeatures2.pNext                = &dynamicRendering;

    vk::DeviceCreateInfo createInfo{};
    createInfo.queueCreateInfoCount    = 1;
    createInfo.pQueueCreateInfos       = &queueCreateInfo;
    createInfo.enabledExtensionCount   = static_cast<uint32_t>(deviceExtensions.size());
    createInfo.ppEnabledExtensionNames = deviceExtensions.data();
    createInfo.pNext                   = &deviceFeatures2;

    auto devRes = physical_device.createDevice(createInfo);
    if (devRes.result != vk::Result::eSuccess)
        return devRes.result;

    device = devRes.value;
    volkLoadDevice(device);
    VULKAN_HPP_DEFAULT_DISPATCHER.init(device);

    queue = device.getQueue(queue_family_index, 0);

    return vk::Result::eSuccess;
}

vk::Result Context::_setup_vma()
{
    vma::VulkanFunctions vulkanFunctions;
    vulkanFunctions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    vulkanFunctions.vkGetDeviceProcAddr   = vkGetDeviceProcAddr;

    vma::AllocatorCreateInfo allocatorInfo{};
    allocatorInfo.flags            = vma::AllocatorCreateFlagBits::eBufferDeviceAddress |
                                     vma::AllocatorCreateFlagBits::eExtMemoryBudget;
    allocatorInfo.vulkanApiVersion = VK_API_VERSION_1_3;
    allocatorInfo.physicalDevice   = physical_device;
    allocatorInfo.device           = device;
    allocatorInfo.instance         = instance;
    allocatorInfo.pVulkanFunctions = &vulkanFunctions;

    auto allocRes = vma::createAllocator(allocatorInfo);
    if (allocRes.result == vk::Result::eSuccess) {
        vma_allocator = allocRes.value;
    }
    return allocRes.result;
}

vk::Result Context::_setup_allocators()
{
    root_pinned_allocator = std::make_unique<PinnedMemoryAllocator>(this);
    root_pool_allocator   = std::make_unique<PoolAllocator>(this);
    pool_allocator        = root_pool_allocator.get();
    pinned_allocator      = root_pinned_allocator.get();
    return vk::Result::eSuccess;
}

vk::Result Context::_setup_compiler()
{
    auto res = ShaderCompiler::create();
    if (res.ok()) {
        shader_compiler = std::move(res.val());
        shader_compiler->set_debug_info(_shader_debug_info);
        return vk::Result::eSuccess;
    }
    return res.err();
}

vk::Result Context::_setup_streams()
{
    (void)create_stream();
    return vk::Result::eSuccess;
}

} // namespace pictor
