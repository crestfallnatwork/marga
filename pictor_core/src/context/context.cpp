#define VMA_IMPLEMENTATION
#include <volk.h>

#include "pictor_core/context/context.hpp"

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

namespace pictor {

Expected<std::unique_ptr<Context>, vk::Result> Context::create() {
  if (volkInitialize() != VK_SUCCESS) {
    return vk::Result::eErrorInitializationFailed;
  }
  VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);

  auto ctx = std::unique_ptr<Context>(new Context());
  vk::Result res = ctx->_init();
  if (res == vk::Result::eSuccess) {
    return ctx;
  }
  return res;
}

Context::~Context() {
  wait_idle();
  _streams.clear();
  pinned_allocator.reset();
  pool_allocator.reset();

  if (device) {
    vma_allocator.destroy();
    device.destroy();
  }
  if (instance) {
    instance.destroy();
  }
}

StreamHandle Context::create_stream() {
  auto handle = static_cast<StreamHandle>(_streams.size());
  _streams.emplace_back(std::make_unique<Stream>(device, queue_family_index));
  return handle;
}

vk::Result Context::submit(StreamHandle handle) {
  Stream *s = get_stream(handle);
  if (s) {
    return s->flush(queue);
  }
  return vk::Result::eErrorInvalidExternalHandle;
}
vk::Result Context::fsync(StreamHandle handle) {
  Stream *s = get_stream(handle);
  if (s) {
    return s->fsync(queue);
  }
  return vk::Result::eErrorInvalidExternalHandle;
}

Stream *Context::get_stream(StreamHandle handle) {
  if (handle < _streams.size()) {
    return _streams[handle].get();
  }
  return nullptr;
}

void Context::sync(StreamHandle handle) {
  Stream *s = get_stream(handle);
  if (s) {
    s->synchronize();
  }
}

void Context::barrier(const AllocationInfo &resource, Access from, Access to,
                      StreamHandle handle) {
  Stream *s = get_stream(handle);
  if (s) {
    s->barrier(resource, from, to);
  }
}

void Context::barrier(const std::vector<ResourceBarrier> &barriers,
                      StreamHandle handle) {
  Stream *s = get_stream(handle);
  if (s) {
    s->barrier(barriers);
  }
}

void Context::copy(const AllocationInfo &src, const AllocationInfo &dst,
                   StreamHandle handle) {
  Stream *s = get_stream(handle);
  if (s) {
    s->_copy(src, dst);
  }
}

void Context::fill(const AllocationInfo &dst, uint32_t value,
                   StreamHandle handle) {
  return fill(dst, dst.offset, dst.size, value);
}

void Context::fill(const AllocationInfo &dst, size_t offset, size_t size,
                   uint32_t value, StreamHandle handle) {
  Stream *s = get_stream(handle);
  if (s) {
    s->_fill(dst, offset, size, value);
  }
}

void Context::update(const AllocationInfo &dst, size_t offset, size_t size,
                     const void *data, StreamHandle handle) {
  Stream *s = get_stream(handle);
  if (s) {
    s->_update(dst, offset, size, data);
  }
}

void Context::deallocate(AllocationInfo &info) { info.alloc->deallocate(info); }
void Context::invalidate(AllocationInfo &info) { info.alloc->invalidate(info); }
void Context::flush(AllocationInfo &info) { info.alloc->flush(info); }

vk::Result Context::_init() {
  vk::Result res = _create_instance();
  if (res != vk::Result::eSuccess)
    return res;

  res = _select_physical_device();
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

vk::Result Context::_create_instance() {
  vk::ApplicationInfo appInfo{};
  appInfo.pApplicationName = "pictor-core";
  appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.pEngineName = "pictor-core";
  appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.apiVersion = VK_API_VERSION_1_3;

  std::vector<const char *> layers = {};
#ifdef PICTOR_DEBUG
  layers.push_back("VK_LAYER_KHRONOS_validation");
#endif

  vk::InstanceCreateInfo createInfo{};
  createInfo.pApplicationInfo = &appInfo;
  createInfo.enabledLayerCount = static_cast<uint32_t>(layers.size());
  createInfo.ppEnabledLayerNames = layers.data();

  auto res = vk::createInstance(createInfo);
  if (res.result == vk::Result::eSuccess) {
    instance = res.value;
    volkLoadInstance(instance);
    VULKAN_HPP_DEFAULT_DISPATCHER.init(instance);
  }
  return res.result;
}

vk::Result Context::_select_physical_device() {
  auto devicesRes = instance.enumeratePhysicalDevices();
  if (devicesRes.result != vk::Result::eSuccess)
    return devicesRes.result;
  if (devicesRes.value.empty())
    return vk::Result::eErrorInitializationFailed;

  auto score_device = [&](vk::PhysicalDevice dev) {
    int score = 0;
    auto props = dev.getProperties();

    if (props.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
      score += 1000;
    } else if (props.deviceType == vk::PhysicalDeviceType::eIntegratedGpu) {
      score += 500;
    }

    bool has_mesh_shader = false;
    auto extensionsRes = dev.enumerateDeviceExtensionProperties();
    if (extensionsRes.result == vk::Result::eSuccess) {
      for (const auto &ext : extensionsRes.value) {
        if (std::string(ext.extensionName) ==
            VK_EXT_MESH_SHADER_EXTENSION_NAME) {
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
  int best_score = -1;

  for (const auto &dev : devicesRes.value) {
    int score = score_device(dev);
    if (score > best_score) {
      best_score = score;
      best_device = dev;
    }
  }

  if (!best_device)
    return vk::Result::eErrorFeatureNotPresent;

  physical_device = best_device;
  return vk::Result::eSuccess;
}

vk::Result Context::_create_logical_device() {
  auto queueFamilies = physical_device.getQueueFamilyProperties();
  int graphicsIndex = -1;

  for (uint32_t i = 0; i < queueFamilies.size(); i++) {
    if ((queueFamilies[i].queueFlags & vk::QueueFlagBits::eGraphics) &&
        (queueFamilies[i].queueFlags & vk::QueueFlagBits::eCompute)) {
      graphicsIndex = i;
      break;
    }
  }

  if (graphicsIndex == -1)
    return vk::Result::eErrorInitializationFailed;

  queue_family_index = static_cast<uint32_t>(graphicsIndex);

  float queuePriority = 1.0f;
  vk::DeviceQueueCreateInfo queueCreateInfo{};
  queueCreateInfo.queueFamilyIndex = queue_family_index;
  queueCreateInfo.queueCount = 1;
  queueCreateInfo.pQueuePriorities = &queuePriority;

  std::vector<const char *> deviceExtensions = {
      VK_EXT_MESH_SHADER_EXTENSION_NAME, VK_KHR_MAINTENANCE_4_EXTENSION_NAME,
      VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,
      VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME
  };
  vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT atomicFloatFeatures;
  atomicFloatFeatures.shaderBufferFloat32AtomicAdd = VK_TRUE;

  vk::PhysicalDeviceMeshShaderFeaturesEXT meshFeatures;
  meshFeatures.meshShader = VK_TRUE;
  meshFeatures.taskShader = VK_TRUE;
  meshFeatures.pNext = &atomicFloatFeatures;

  vk::PhysicalDeviceBufferDeviceAddressFeatures bdaFeatures;
  bdaFeatures.bufferDeviceAddress = VK_TRUE;
  bdaFeatures.pNext = &meshFeatures;

  vk::PhysicalDeviceSynchronization2Features sync2Features;
  sync2Features.synchronization2 = VK_TRUE;
  sync2Features.pNext = &bdaFeatures;

  vk::PhysicalDeviceMaintenance4Features maintenance4;
  maintenance4.maintenance4 = VK_TRUE;
  maintenance4.pNext = &sync2Features;

  vk::PhysicalDeviceDynamicRenderingFeatures dynamicRendering;
  dynamicRendering.dynamicRendering = VK_TRUE;
  dynamicRendering.pNext = &maintenance4;

  vk::PhysicalDeviceFeatures2 deviceFeatures2;
  deviceFeatures2.features.shaderInt64 = VK_TRUE;
  deviceFeatures2.pNext = &dynamicRendering;

  vk::PhysicalDeviceSubgroupProperties subgroupProps;
  vk::PhysicalDeviceProperties2 devProps2;
  devProps2.pNext = &subgroupProps;
  physical_device.getProperties2(&devProps2);
  subgroup_size = subgroupProps.subgroupSize;

  vk::DeviceCreateInfo createInfo{};
  createInfo.queueCreateInfoCount = 1;
  createInfo.pQueueCreateInfos = &queueCreateInfo;
  createInfo.enabledExtensionCount =
      static_cast<uint32_t>(deviceExtensions.size());
  createInfo.ppEnabledExtensionNames = deviceExtensions.data();
  createInfo.pNext = &deviceFeatures2;

  auto devRes = physical_device.createDevice(createInfo);
  if (devRes.result != vk::Result::eSuccess)
    return devRes.result;

  device = devRes.value;
  volkLoadDevice(device);
  VULKAN_HPP_DEFAULT_DISPATCHER.init(device);

  queue = device.getQueue(queue_family_index, 0);

  return vk::Result::eSuccess;
}

vk::Result Context::_setup_vma() {
  vma::VulkanFunctions vulkanFunctions;
  vulkanFunctions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
  vulkanFunctions.vkGetDeviceProcAddr = vkGetDeviceProcAddr;

  vma::AllocatorCreateInfo allocatorInfo{};
  allocatorInfo.flags = vma::AllocatorCreateFlagBits::eBufferDeviceAddress;
  allocatorInfo.vulkanApiVersion = VK_API_VERSION_1_3;
  allocatorInfo.physicalDevice = physical_device;
  allocatorInfo.device = device;
  allocatorInfo.instance = instance;
  allocatorInfo.pVulkanFunctions = &vulkanFunctions;

  auto allocRes = vma::createAllocator(allocatorInfo);
  if (allocRes.result == vk::Result::eSuccess) {
    vma_allocator = allocRes.value;
  }
  return allocRes.result;
}

vk::Result Context::_setup_allocators() {
  pinned_allocator = std::make_unique<PinnedMemoryAllocator>(this);
  pool_allocator = std::make_unique<PoolAllocator>(this);
  return vk::Result::eSuccess;
}

vk::Result Context::_setup_compiler() {
  auto res = ShaderCompiler::create();
  if (res.ok()) {
    shader_compiler = std::move(res.val());
    return vk::Result::eSuccess;
  }
  return res.err();
}

vk::Result Context::_setup_streams() {
  (void)create_stream();
  return vk::Result::eSuccess;
}

} // namespace pictor
