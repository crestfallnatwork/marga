#include "clipp.h"
#include "csr.hpp"
#include "pagerank.hpp"
#include "pagerank_validation.hpp"
#include "pictor_core/context/context.hpp"

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

    std::cout << "Importing graph as CSR: " << graph_file << "..." << std::endl;
    auto g = pictor::graphs::Graph::import(ctx, graph_file, false,
                                           pictor::graphs::GraphLayout::CSR);
    pictor::graphs::PageRank pr(ctx);
    pr.run(g, 1e-5, pictor::graphs::GraphLayout::CSR);

    // std::cout << "Importing graph as CSC: " << graph_file << "..." << std::endl;
    // auto g = pictor::graphs::Graph::import(ctx, graph_file, false,
    //                                        pictor::graphs::GraphLayout::CSC);
    // pictor::graphs::PageRank pr(ctx);
    // pr.run(g, 1e-5, pictor::graphs::GraphLayout::CSC);

    // std::cout << "Running GPU PageRank (20 iterations)..." << std::endl;
    // auto [ranks, timings] = pr.run(g, 20);

    // std::cout << "\n=== PageRank Results ===" << std::endl;
    // std::cout << "  Preprocessing (Out-degrees): " <<
    // timings.preprocessing_ms << " ms" << std::endl; std::cout << "
    // Initialization (H2D):        " << timings.init_ms << " ms" << std::endl;
    // std::cout << "  Average Iteration Time:      " <<
    // timings.iteration_avg_ms << " ms" << std::endl; std::cout << "  Total GPU
    // Execution Time:    " << timings.total_ms << " ms" << std::endl;

    // if (validate) {
    //     std::cout << "\nRunning CPU Validation..." << std::endl;
    //     auto [cpu_ranks, cpu_t] = pictor::graphs::PageRankValidator::run(g,
    //     20); std::cout << "  CPU Total: " << cpu_t.traversal_ms << " ms" <<
    //     std::endl;

    //     pictor::graphs::PageRankValidator::verify(g, ranks, cpu_ranks);
    // }

    // // Print top 5 pageranks for validation
    // std::cout << "\nTop 5 GPU PageRanks (Samples):" << std::endl;
    // for (int i = 0; i < std::min(5, (int)g.num_vertices); i++) {
    //     std::cout << "  Vertex " << i << ": " << ranks[i] << std::endl;
    // }

  } else {
    std::cout << make_man_page(cli, argv[0]);
    return 1;
  }

  return 0;
}
