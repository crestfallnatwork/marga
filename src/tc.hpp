#include "tc.shader.h"
namespace pictor {
struct Context;
namespace graphs {
struct Graph;

struct TC {
    TC(Context &ctx);
    ~TC();
    void init_buffers();
    uint64_t run(Graph const &g);

    using TCExec = tc_shader::TCExecutor;
    using TCIn = tc_shader::TCIn;
    std::unique_ptr<TCExec> tc;

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
