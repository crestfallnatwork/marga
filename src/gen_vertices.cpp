// SPDX-License-Identifier: GPL-2.0-only
// Author: Himanshu Gupta <himanshu3@iisc.ac.in>

#include "clipp.h"
#include "csr.hpp"
#include "pictor_core/context/context.hpp"
#include "pictor_core/memory/heap_allocator.hpp"

#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>

int main(int argc, char* argv[])
{
    using namespace clipp;

    std::string graph_file;
    uint32_t num_vertices = 64;

    auto cli = (value("graph file", graph_file),
                (option("-n", "--num-vertices") & value("number of vertices", num_vertices)) %
                     "number of valid source vertices to generate (default 64)");

    if (parse(argc, argv, cli)) {
        auto ctx_res = pictor::Context::create();
        if (!ctx_res.ok()) {
            std::cerr << "Failed to create context" << std::endl;
            return 1;
        }
        auto& ctx = *ctx_res.val();

        auto g = pictor::graphs::Graph::import(ctx, graph_file);

        std::vector<uint32_t> valid_sources;
        
        // Random number generation setup
        std::srand(42); // fixed seed for reproducibility
        
        while (valid_sources.size() < num_vertices) {
            // Generate a 64-bit random number to cover > RAND_MAX vertices if needed
            uint64_t r1 = std::rand();
            uint64_t r2 = std::rand();
            uint32_t v = (r1 ^ (r2 << 15)) % g.num_vertices;
            
            uint64_t edges = g.row_offsets[v + 1] - g.row_offsets[v];
            if (edges > 0) {
                valid_sources.push_back(v);
            }
        }
        
        for (uint32_t src : valid_sources) {
            std::cout << src << "\n";
        }
    } else {
        std::cerr << make_man_page(cli, argv[0]);
        return 1;
    }

    return 0;
}
