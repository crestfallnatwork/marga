#include "pictor_core/context/context.hpp"
#include "pictor_core/execution/mesh_executor.hpp"
#include "pictor_core/execution/compute_executor.hpp"
#include "pictor_core/execution/shader_compiler.hpp"
#include <iostream>
#include <vector>
#include <iomanip>
#include <cstring>

const char* slangSource = R"(
    struct AddConstants {
        uint64_t a_addr;
        uint64_t b_addr;
        uint64_t c_addr;
        uint count;
    };

    [shader("compute")]
    [numthreads(64, 1, 1)]
    void main(uint3 threadId : SV_DispatchThreadID, uniform AddConstants pc) {
        if (threadId.x < pc.count) {
            float* a = (float*)pc.a_addr;
            float* b = (float*)pc.b_addr;
            float* c = (float*)pc.c_addr;
            c[threadId.x] = a[threadId.x] + b[threadId.x];
        }
    }
)";

struct AddConstants {
    uint64_t a_addr;
    uint64_t b_addr;
    uint64_t c_addr;
    uint32_t count;
};

struct MulConstants {
    uint64_t res_addr;
    uint64_t c_addr;
    uint64_t out_addr;
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

    // --- Slang JIT Test ---
    std::cout << "Testing Slang JIT..." << std::endl;
    auto compilerRes = pictor::ShaderCompiler::create();
    if (std::holds_alternative<vk::Result>(compilerRes)) {
        std::cerr << "Failed to create compiler." << std::endl;
        return 1;
    }
    auto& compiler = *std::get<std::unique_ptr<pictor::ShaderCompiler>>(compilerRes);
    auto compileResult = compiler.compile(slangSource, "main", pictor::ShaderStage::Compute);
    if (!compileResult.success) {
        std::cerr << "Slang Compilation Failed: " << compileResult.diagnostics << std::endl;
        return 1;
    }
    std::cout << "Slang JIT Compilation SUCCESS!" << std::endl;

    // --- Create Executors ---
    auto addRes = pictor::ComputeExecutor<AddConstants>::create(ctx, "../test/shaders/add.comp.spv");
    auto mulRes = pictor::MeshExecutor<MulConstants>::create(ctx, "../test/shaders/mul.task.spv", "../test/shaders/mul.mesh.spv");

    if (std::holds_alternative<vk::Result>(addRes) || std::holds_alternative<vk::Result>(mulRes)) {
        std::cerr << "Failed to create executors." << std::endl;
        return 1;
    }
    auto& addExec = *std::get<std::unique_ptr<pictor::ComputeExecutor<AddConstants>>>(addRes);
    auto& mulExec = *std::get<std::unique_ptr<pictor::MeshExecutor<MulConstants>>>(mulRes);

    const uint32_t count = 1024;
    const size_t size = count * sizeof(float);

    // --- Allocate All Resources Upfront ---
    auto bufA = ctx.pool_allocator->allocate(size, 0);
    auto bufB = ctx.pool_allocator->allocate(size, 0);
    auto bufC = ctx.pool_allocator->allocate(size, 0); // result of add
    auto bufD = ctx.pool_allocator->allocate(size, 0); // used for multiplication
    auto bufOut = ctx.pool_allocator->allocate(size, 0); // final result

    auto hostA = ctx.pinned_allocator->allocate(size, 0);
    auto hostB = ctx.pinned_allocator->allocate(size, 0);
    auto hostD = ctx.pinned_allocator->allocate(size, 0);
    auto hostFinal = ctx.pinned_allocator->allocate(size, 0);

    // Initialize data
    float* dataA = static_cast<float*>(hostA.mappedPtr);
    float* dataB = static_cast<float*>(hostB.mappedPtr);
    float* dataD = static_cast<float*>(hostD.mappedPtr);
    for (uint32_t i = 0; i < count; ++i) {
        dataA[i] = 10.0f;
        dataB[i] = 20.0f;
        dataD[i] = 2.0f;
    }
    ctx.pinned_allocator->flush(hostA);
    ctx.pinned_allocator->flush(hostB);
    ctx.pinned_allocator->flush(hostD);

    pictor::StreamHandle stream = ctx.create_stream();

    // 1. Send all data to GPU upfront
    ctx.copy(hostA, bufA, stream);
    ctx.copy(hostB, bufB, stream);
    ctx.copy(hostD, bufD, stream);

    // 2. Barrier: Transfer -> Compute Read
    ctx.barrier(bufA, pictor::Access::TransferWrite, pictor::Access::Read, stream);
    ctx.barrier(bufB, pictor::Access::TransferWrite, pictor::Access::Read, stream);
    ctx.barrier(bufD, pictor::Access::TransferWrite, pictor::Access::Read, stream);

    // 3. Chain Part 1: Compute Addition (C = A + B)
    std::cout << "Executing Compute Addition (C = A + B)..." << std::endl;
    AddConstants addPc = { bufA.address, bufB.address, bufC.address, count };
    addExec.record((count + 63) / 64, 1, 1, addPc, ctx, stream);

    // 4. Chain Part 2 Barrier: Compute Write -> Mesh Read
    // C is result of compute, used as source in mesh.
    ctx.barrier(bufC, pictor::Access::Write, pictor::Access::Read, stream);

    // 5. Chain Part 3: Mesh Multiplication (Out = C * D)
    std::cout << "Executing Mesh Multiplication (Out = C * D)..." << std::endl;
    MulConstants mulPc = { bufC.address, bufD.address, bufOut.address, count };
    mulExec.record(1, 1, 1, mulPc, ctx, stream);

    // 6. Chain Part 4 Barrier: Mesh Write -> Transfer Read
    ctx.barrier(bufOut, pictor::Access::Write, pictor::Access::TransferRead, stream);

    // 7. Copy Final Result back to Host
    ctx.copy(bufOut, hostFinal, stream);

    // 8. Flush and Sync Everything
    ctx.flush(stream);
    ctx.sync(stream);

    // --- Verify ---
    ctx.pinned_allocator->invalidate(hostFinal);
    float* resultData = static_cast<float*>(hostFinal.mappedPtr);
    uint32_t errors = 0;
    for (uint32_t i = 0; i < count; ++i) {
        float expected = (10.0f + 20.0f) * 2.0f; // (A + B) * D = 60.0
        if (std::abs(resultData[i] - expected) > 1e-5f) {
            if (errors < 5) std::cerr << "Mismatch at " << i << ": exp " << expected << " got " << resultData[i] << std::endl;
            errors++;
        }
    }

    if (errors == 0) {
        std::cout << "CHAIN SUCCESS: (A + B) * D = 60.0 verified across all elements!" << std::endl;
    } else {
        std::cerr << "CHAIN FAILURE: " << errors << " errors found." << std::endl;
    }

    // Cleanup
    ctx.pool_allocator->deallocate(bufA); ctx.pool_allocator->deallocate(bufB);
    ctx.pool_allocator->deallocate(bufC); ctx.pool_allocator->deallocate(bufD);
    ctx.pool_allocator->deallocate(bufOut);
    ctx.pinned_allocator->deallocate(hostA); ctx.pinned_allocator->deallocate(hostB);
    ctx.pinned_allocator->deallocate(hostD); ctx.pinned_allocator->deallocate(hostFinal);

    return errors == 0 ? 0 : 1;
}
