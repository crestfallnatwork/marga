// SPDX-License-Identifier: GPL-2.0-only
// Author: Himanshu Gupta <himanshu3@iisc.ac.in>

#include "clipp.h"
#include "csr.hpp"
#include "pagerank.hpp"
#include "pictor_core/context/context.hpp"

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char *argv[]) {
  using namespace clipp;

  std::string graph_file;
  bool validate = false;

  auto cli =
      group(value("graph file", graph_file) % "MTX graph file to process",
            option("-v", "--validate").set(validate) %
                "validate GPU results against parallel CPU PageRank");

  if (parse(argc, argv, cli)) {
    auto ctx_res = pictor::Context::create();
    if (!ctx_res.ok()) {
      std::cerr << "Failed to create context" << std::endl;
      return 1;
    }
    auto &ctx = *ctx_res.val();

    std::vector<float> pr_gpu, pr_cpu;

    {
      std::cout << "Importing graph as CSC: " << graph_file << "..."
                << std::endl;
      auto g = pictor::graphs::Graph::import(ctx, graph_file, false,
                                             pictor::graphs::GraphLayout::CSC);
      pictor::graphs::PageRank pr(ctx);
      auto [pr_res, _] = pr.run(g, 1e-4, pictor::graphs::GraphLayout::CSC);
      pr_gpu = std::move(pr_res);
    }

    if (validate) {
      std::cout << "Running sequential push iteration to validate fixed point "
                   "behaviour:"
                << std::endl;
      auto g = pictor::graphs::Graph::import(ctx, graph_file, false,
                                             pictor::graphs::GraphLayout::CSR);
      pr_cpu = pr_gpu;
      float dangling = 0.;
      float all_error = 0.;
      std::fill(pr_cpu.begin(), pr_cpu.end(), 0.);
      for (uint32_t vertex = 0; vertex < g.num_vertices; vertex++) {
        uint64_t edges = g.row_offsets[vertex + 1] - g.row_offsets[vertex];
        for (uint64_t dst :
             g.column_indices.subspan(g.row_offsets[vertex], edges)) {
          pr_cpu[dst] += (pr_gpu[vertex] / edges);
        }
      }
      for (uint32_t vertex = 0; vertex < g.num_vertices; vertex++) {
        uint64_t edges = g.row_offsets[vertex + 1] - g.row_offsets[vertex];
        if (edges == 0) {
          dangling += pr_gpu[vertex];
        }
      }
      for (uint32_t vertex = 0; vertex < g.num_vertices; vertex++) {
        auto csr_v = ((1. - 0.85) / float(g.num_vertices)) +
                     0.85 * (pr_cpu[vertex] + (dangling / g.num_vertices));
        auto err = std::abs(csr_v - pr_gpu[vertex]);
        all_error += err;
        if (err > 1e-4) {
          std::cout
              << std::format(
                     "Didn't converge, vertex: {}, csc val: {}, new val: {}",
                     vertex, pr_gpu[vertex], csr_v)
              << std::endl;
        }
      }
      if (all_error > 1e-4) {
        std::cout << std::format("Didn't converge, error: {}\n", all_error);
      }
    }

  } else {
    std::cout << make_man_page(cli, argv[0]);
    return 1;
  }

  return 0;
}
