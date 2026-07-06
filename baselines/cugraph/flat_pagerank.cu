#include "csr.hpp"
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

using vertex_t = int32_t;
using edge_t   = int32_t;
using weight_t = float;

__global__ void compute_out_degrees_kernel(const vertex_t* __restrict__ indices, edge_t num_edges,
                                           uint32_t* __restrict__ out_degrees)
{
    edge_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_edges) {
        vertex_t src = indices[tid];
        atomicAdd(&out_degrees[src], 1);
    }
}

__global__ void init_kernel(float* past_pr, float* curr_pr, const uint32_t* out_degrees,
                            vertex_t num_vertices)
{
    vertex_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < num_vertices) {
        float init_val = 1.0f / num_vertices;
        if (out_degrees[v] > 0)
            init_val /= out_degrees[v];
        past_pr[v] = init_val;
        curr_pr[v] = 0.0f;
    }
}

__global__ void edge_pull_kernel(const edge_t* __restrict__ row_offsets,
                                 const vertex_t* __restrict__ column_indices,
                                 const float* __restrict__ past_pr, float* __restrict__ curr_pr,
                                 vertex_t num_vertices, edge_t num_edges)
{
    edge_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_edges)
        return;

    vertex_t left  = 0;
    vertex_t right = num_vertices - 1;
    vertex_t dst   = 0;
    while (left <= right) {
        vertex_t mid = left + (right - left) / 2;
        if (row_offsets[mid] <= tid) {
            dst  = mid;
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    vertex_t src = column_indices[tid];
    float pull   = past_pr[src];

    atomicAdd(&curr_pr[dst], pull);
}

__global__ void dangling_kernel(const float* __restrict__ past_pr,
                                const uint32_t* __restrict__ out_degrees,
                                float* __restrict__ dangling_sum, vertex_t num_vertices)
{
    vertex_t v      = blockIdx.x * blockDim.x + threadIdx.x;
    float local_sum = 0.0f;
    if (v < num_vertices) {
        if (out_degrees[v] == 0) {
            local_sum = past_pr[v];
        }
    }

    // Warp-level reduction (optional, but let's just do atomicAdd for simplicity since it's just a
    // test) Actually atomicAdd on a single global float from all threads is very slow. Let's do
    // warp reduce.
    unsigned mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(mask, local_sum, offset);
    }
    if (threadIdx.x % 32 == 0) {
        atomicAdd(dangling_sum, local_sum);
    }
}

__global__ void reduce_kernel(float* __restrict__ past_pr, float* __restrict__ curr_pr,
                              const uint32_t* __restrict__ out_degrees,
                              float* __restrict__ delta_sum, float dangling_total,
                              vertex_t num_vertices)
{
    vertex_t v        = blockIdx.x * blockDim.x + threadIdx.x;
    float local_delta = 0.0f;

    if (v < num_vertices) {
        float dangling = dangling_total / num_vertices;
        float cur_pr   = curr_pr[v];
        cur_pr         = (1.0f - 0.85f) / num_vertices + 0.85f * (cur_pr + dangling);

        uint32_t out_deg = out_degrees[v];
        float old_pr     = past_pr[v];
        if (out_deg > 0)
            old_pr *= out_deg;

        local_delta = fabsf(old_pr - cur_pr);

        float pull_val = cur_pr;
        if (out_deg > 0)
            pull_val /= out_deg;

        curr_pr[v] = pull_val;
        past_pr[v] = 0.0f; // Clear for next iteration
    }

    unsigned mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset /= 2) {
        local_delta += __shfl_down_sync(mask, local_delta, offset);
    }
    if (threadIdx.x % 32 == 0) {
        atomicAdd(delta_sum, local_delta);
    }
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <graph.mtx>\n";
        return 1;
    }

    std::string filename = argv[1];
    std::cout << "Importing graph as CSC: " << filename << "...\n";
    auto h_graph = Graph<vertex_t, edge_t>::import(filename, false, GraphLayout::CSC);

    edge_t* d_row_offsets;
    vertex_t* d_column_indices;
    cudaMalloc(&d_row_offsets, h_graph.row_offsets.size() * sizeof(edge_t));
    cudaMalloc(&d_column_indices, h_graph.column_indices.size() * sizeof(vertex_t));
    cudaMemcpy(d_row_offsets, h_graph.row_offsets.data(),
               h_graph.row_offsets.size() * sizeof(edge_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_column_indices, h_graph.column_indices.data(),
               h_graph.column_indices.size() * sizeof(vertex_t), cudaMemcpyHostToDevice);

    uint32_t* d_out_degrees;
    cudaMalloc(&d_out_degrees, h_graph.num_vertices * sizeof(uint32_t));
    cudaMemset(d_out_degrees, 0, h_graph.num_vertices * sizeof(uint32_t));

    int blockSize  = 256;
    int numBlocksE = (h_graph.num_edges + blockSize - 1) / blockSize;
    int numBlocksV = (h_graph.num_vertices + blockSize - 1) / blockSize;

    compute_out_degrees_kernel<<<numBlocksE, blockSize>>>(d_column_indices, h_graph.num_edges,
                                                          d_out_degrees);
    cudaDeviceSynchronize();

    float *d_past_pr, *d_curr_pr, *d_dangling_sum, *d_delta_sum;
    cudaMalloc(&d_past_pr, h_graph.num_vertices * sizeof(float));
    cudaMalloc(&d_curr_pr, h_graph.num_vertices * sizeof(float));
    cudaMalloc(&d_dangling_sum, sizeof(float));
    cudaMalloc(&d_delta_sum, sizeof(float));

    float target_delta = 1e-4f;
    float h_delta      = 1.0f;
    int iterations     = 0;

    init_kernel<<<numBlocksV, blockSize>>>(d_past_pr, d_curr_pr, d_out_degrees,
                                           h_graph.num_vertices);
    cudaDeviceSynchronize();

    float* curr = d_curr_pr;
    float* past = d_past_pr;

    auto t1 = std::chrono::high_resolution_clock::now();

    do {
        cudaMemset(d_dangling_sum, 0, sizeof(float));
        cudaMemset(d_delta_sum, 0, sizeof(float));

        edge_pull_kernel<<<numBlocksE, blockSize>>>(d_row_offsets, d_column_indices, past, curr,
                                                    h_graph.num_vertices, h_graph.num_edges);

        dangling_kernel<<<numBlocksV, blockSize>>>(past, d_out_degrees, d_dangling_sum,
                                                   h_graph.num_vertices);

        float h_dangling;
        cudaMemcpy(&h_dangling, d_dangling_sum, sizeof(float), cudaMemcpyDeviceToHost);

        reduce_kernel<<<numBlocksV, blockSize>>>(past, curr, d_out_degrees, d_delta_sum, h_dangling,
                                                 h_graph.num_vertices);

        cudaMemcpy(&h_delta, d_delta_sum, sizeof(float), cudaMemcpyDeviceToHost);

        std::swap(curr, past);
        iterations++;

    } while (h_delta > target_delta);

    auto t2     = std::chrono::high_resolution_clock::now();
    float times = std::chrono::duration<float, std::milli>(t2 - t1).count();

    std::cout << "Last delta: " << h_delta << "\n";
    std::cout << "Iterations: " << iterations << ", Time: " << times
              << "ms, Time/Iteration: " << times / iterations << "ms\n";

    return 0;
}
