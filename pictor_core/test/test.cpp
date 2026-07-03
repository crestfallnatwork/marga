#include <pictor_core/context/context.hpp>
#include <pictor_core/memory/heap_allocator.hpp>

#include <chrono>
#include <iostream>
#include <locale>

using namespace pictor;
using namespace std::chrono;
using namespace std::chrono_literals;

int main()
{
    auto ctx_res = Context::create();
    if (!ctx_res.ok()) {
        std::cerr << "cannot create context" << std::endl;
        return -1;
    }
    auto ctx       = std::move(ctx_res.val());
    auto dev_mem   = ctx->get_device_free_memory();
    auto dev_mem_9 = size_t(dev_mem * 0.9);

    auto t_dev_alloc_init  = 0.ms;
    auto t_host_alloc_init = 0.ms;
    auto t_dev_allocation  = 0.ms;
    auto t_host_allocation = 0.ms;

    auto t1                 = std::chrono::high_resolution_clock::now();
    auto dev_allocator_res  = HeapAllocator::create(ctx.get(), ctx->pool_allocator, dev_mem_9);
    auto t2                 = std::chrono::high_resolution_clock::now();
    t_dev_alloc_init        = t2 - t1;
    t1                      = std::chrono::high_resolution_clock::now();
    auto host_allocator_res = HeapAllocator::create(ctx.get(), ctx->pinned_allocator, dev_mem_9);
    t2                      = std::chrono::high_resolution_clock::now();
    t_host_alloc_init       = t2 - t1;
    if (!dev_allocator_res.ok()) {
        std::cout << std::format("Create Dev Allocator Err: {}\n",
                                 vk::to_string(dev_allocator_res.err()));
        return -1;
    }
    if (!host_allocator_res.ok()) {
        std::cout << std::format("Create Host Allocator Err: {}\n",
                                 vk::to_string(host_allocator_res.err()));
        return -1;
    }
    auto dev_allocator  = std::move(dev_allocator_res.val());
    auto host_allocator = std::move(host_allocator_res.val());

    for(int i = 0; i < 100; i++) {
        t1                = std::chrono::high_resolution_clock::now();
        auto alloc_dev    = dev_allocator->allocate(32, 1);
        t2                = std::chrono::high_resolution_clock::now();
        t_dev_allocation  += (t2 - t1);
        t1                = std::chrono::high_resolution_clock::now();
        auto alloc_host   = host_allocator->allocate(32, 1);
        t2                = std::chrono::high_resolution_clock::now();
        t_host_allocation += (t2 - t1);
    }

    std::cout << std::format(std::locale("en_US.UTF-8"),
                             "allocated device memory: {:L}, host memory: {:L}", dev_mem_9,
                             dev_mem_9)
              << std::endl;
    std::cout << std::format("creating allocators took: {}(dev), {}(host)\n", t_dev_alloc_init,
                             t_host_alloc_init);
    std::cout << std::format("allocation took: {}(dev), {}(host)\n", t_dev_allocation/100,
                             t_host_allocation/100);
}
