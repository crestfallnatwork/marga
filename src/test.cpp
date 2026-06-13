#include "pictor_core/context/context.hpp"
#include "pictor_core/execution/mesh_executor.hpp"
#include <iostream>
#include <vector>
#include <iomanip>

struct AddConstants {
    uint64_t a_addr;
    uint64_t b_addr;
    uint64_t c_addr;
    uint32_t count;
};

int main() {
    auto ctxRes = pictor::Context::create();
    if (std::holds_alternative<vk::Result>(ctxRes)) {
        std::cerr << "Failed to create context: " << vk::to_string(std::get<vk::Result>(ctxRes)) << std::endl;
        return 1;
    }

    auto& ctx = *std::get<std::unique_ptr<pictor::Context>>(ctxRes);
    std::cout << "Vulkan Context created successfully." << std::endl;

    // Load MeshExecutor
    auto execRes = pictor::MeshExecutor<AddConstants>::create(
        ctx,
        "../test/shaders/add.task.spv",
        "../test/shaders/add.mesh.spv"
    );

    if (std::holds_alternative<vk::Result>(execRes)) {
        std::cerr << "Failed to create executor: " << vk::to_string(std::get<vk::Result>(execRes)) << std::endl;
        return 1;
    }
    auto& executor = *std::get<std::unique_ptr<pictor::MeshExecutor<AddConstants>>>(execRes);

    const uint32_t count = 1024;
    const size_t size = count * sizeof(float);

    // Allocate Pool Buffers
    auto bufA = ctx.pool_allocator->allocate(size, 0);
    auto bufB = ctx.pool_allocator->allocate(size, 0);
    auto bufC = ctx.pool_allocator->allocate(size, 0);

    // Allocate Pinned Buffers
    auto hostA = ctx.pinned_allocator->allocate(size, 0);
    auto hostB = ctx.pinned_allocator->allocate(size, 0);
    auto hostC = ctx.pinned_allocator->allocate(size, 0);

    // Initialize Host Data
    float* dataA = static_cast<float*>(hostA.mappedPtr);
    float* dataB = static_cast<float*>(hostB.mappedPtr);
    for (uint32_t i = 0; i < count; ++i) {
        dataA[i] = static_cast<float>(i);
        dataB[i] = static_cast<float>(i * 2);
    }
    
    ctx.pinned_allocator->flush(hostA);
    ctx.pinned_allocator->flush(hostB);

    // Execution
    pictor::StreamHandle stream = ctx.create_stream();

    // 1. Copy to Device
    ctx.copy(hostA, bufA, stream);
    ctx.copy(hostB, bufB, stream);

    // 2. Barriers: TransferWrite -> Read
    ctx.barrier(bufA, pictor::Access::TransferWrite, pictor::Access::Read, stream);
    ctx.barrier(bufB, pictor::Access::TransferWrite, pictor::Access::Read, stream);

    // 3. Dispatch
    AddConstants pc = { bufA.address, bufB.address, bufC.address, count };
    executor.record(1, 1, 1, pc, ctx, stream);

    // 4. Barrier: Write -> TransferRead
    ctx.barrier(bufC, pictor::Access::Write, pictor::Access::TransferRead, stream);

    // 5. Copy back to Host
    ctx.copy(bufC, hostC, stream);

    // 6. Flush and Sync
    ctx.flush(stream);
    ctx.sync(stream);

    // Verify Result
    ctx.pinned_allocator->invalidate(hostC);
    float* resultData = static_cast<float*>(hostC.mappedPtr);
    
    uint32_t errors = 0;
    for (uint32_t i = 0; i < count; ++i) {
        float expected = static_cast<float>(i + i * 2);
        if (std::abs(resultData[i] - expected) > 1e-5f) {
            if (errors < 10) {
                std::cerr << "Mismatch at index " << i << ": expected " << expected << ", got " << resultData[i] << std::endl;
            }
            errors++;
        }
    }

    if (errors == 0) {
        std::cout << "SUCCESS: All " << count << " elements added correctly!" << std::endl;
    } else {
        std::cerr << "FAILURE: " << errors << " mismatches found." << std::endl;
    }

    // Cleanup
    ctx.pool_allocator->deallocate(bufA);
    ctx.pool_allocator->deallocate(bufB);
    ctx.pool_allocator->deallocate(bufC);
    ctx.pinned_allocator->deallocate(hostA);
    ctx.pinned_allocator->deallocate(hostB);
    ctx.pinned_allocator->deallocate(hostC);

    return errors == 0 ? 0 : 1;
}
