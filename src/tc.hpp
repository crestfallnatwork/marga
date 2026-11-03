#include "tc.shader.h"
namespace pictor {
struct Context;
namespace graphs {
struct Graph;

struct TC {
    TC(Context &ctx, uint32_t cull = 2, uint64_t batch_edges = 1u << 24);
    ~TC();
    void init_buffers();
    uint64_t run(Graph const &g);

    using TCExec = tc_shader::TCExecutor;
    using TCIn = tc_shader::TCIn;
    std::unique_ptr<TCExec> tc;

    // Pruning level: 0 = none, 1 = orientation cull, 2 = orientation +
    // bounded lists (default).
    uint32_t cull;
    // Edges per batch. The per-batch triangle accumulator is 32-bit, so arms
    // whose batch count would exceed 2^32 use a smaller window.
    uint64_t batch_edges;

    // using EdgeSumExec = tc_shader::EdgesumComputeExecutor;
    // using EdgeSumIn = tc_shader::EdgeSumIn;
    // std::unique_ptr<EdgeSumExec> edgeSum;
    
    AllocationInfo result;
    AllocationInfo h_result;

private:
    Context &ctx;
};


} // namespace graphs
} // namespace pictor
