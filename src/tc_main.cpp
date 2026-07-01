// SPDX-License-Identifier: GPL-2.0-only
// Author: Himanshu Gupta <himanshu3@iisc.ac.in>

#include "clipp.h"
#include "csr.hpp"
#include "pictor_core/context/context.hpp"
#include "tc.hpp"

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char* argv[])
{
    using namespace clipp;

    std::string graph_file;
    bool validate = false;

    auto cli = group(value("graph file", graph_file) % "MTX graph file to process",
                     option("-v", "--validate").set(validate) %
                         "validate GPU results against parallel CPU PageRank");

    if (parse(argc, argv, cli)) {
        auto ctx_res = pictor::Context::create();
        if (!ctx_res.ok()) {
            std::cerr << "Failed to create context" << std::endl;
            return 1;
        }
        auto& ctx = *ctx_res.val();

        uint64_t traingles_gpu = 0;

        std::cout << "Importing graph as CSR: " << graph_file << "..." << std::endl;
        auto g =
            pictor::graphs::Graph::import(ctx, graph_file, true, pictor::graphs::GraphLayout::CSR);
        pictor::graphs::TC tc(ctx);
        tc.run(g);
        if (validate) {}

    } else {
        std::cout << make_man_page(cli, argv[0]);
        return 1;
    }

    return 0;
}
