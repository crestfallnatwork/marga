#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include "tc/tc_runner.h"

using namespace marga_abl;

static void print_usage(const char* argv0)
{
    std::printf(
        "usage: %s --graph <path.mtx> [options]\n"
        "\n"
        "One thread per edge, no task/mesh decomposition; the cull level selects\n"
        "MARGA's pruning (orientation cull + bounded lists).\n"
        "\n"
        "options:\n"
        "  --cull none|orient|bounded  pruning level (default bounded)\n"
        "                           none   = all edges, full-list intersection, count/6\n"
        "                           orient = u<v edges only, full-list intersection, count/3\n"
        "                           bounded= u<v edges, bounded lists, count once (MARGA)\n"
        "  --preorient              orient the dataset in a two-pass device preprocessing step\n"
        "                           (per-vertex count + prefix sum, then write the u<v edge\n"
        "                           list) and run the arm on the oriented list; no load\n"
        "                           balancing. The pre-pass is timed separately (prepass_ms).\n"
        "  --batch N                edges per launch (default 16777216 = 2^24, MARGA's TC\n"
        "                           batch size, host round-trip per batch); 0 = whole graph\n"
        "  --reps N                 timed repetitions (default 3)\n"
        "  --warmup N               warmup repetitions (default 1)\n"
        "  --validate               compare the triangle count against the CPU reference\n"
        "  --no-cache               ignore/do not write the .sym.csrbin cache\n"
        "  --csv <path>             append one result row to a CSV file\n"
        "\n"
        "The graph is always loaded symmetrized with sorted, deduplicated adjacency\n"
        "(MARGA's TC import semantics); the symmetrized CSR is cached as <graph>.sym.csrbin.\n",
        argv0);
}

int main(int argc, char** argv)
{
    TcAblationConfig cfg;
    std::string     graph_path;
    std::string     csv_path;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need             = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", name);
                std::exit(1);
            }
            return argv[++i];
        };
        if (arg == "--graph")
            graph_path = need("--graph");
        else if (arg == "--cull")
            cfg.cull = need("--cull");
        else if (arg == "--batch")
            cfg.batch = std::stoull(need("--batch"));
        else if (arg == "--reps")
            cfg.reps = std::stoi(need("--reps"));
        else if (arg == "--warmup")
            cfg.warmup = std::stoi(need("--warmup"));
        else if (arg == "--preorient")
            cfg.preorient = true;
        else if (arg == "--validate")
            cfg.validate = true;
        else if (arg == "--no-cache")
            cfg.use_cache = false;
        else if (arg == "--csv")
            csv_path = need("--csv");
        else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }

    if (graph_path.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    try {
        const auto t0 = std::chrono::high_resolution_clock::now();
        const HostGraph g = pictor::graphs::import_mtx(
            graph_path, /*symmetrize=*/true, pictor::graphs::GraphLayout::CSR, cfg.use_cache);
        const auto t1 = std::chrono::high_resolution_clock::now();
        const double load_s = std::chrono::duration<double>(t1 - t0).count();
        std::printf("graph=%s V=%" PRIu64 " E=%" PRIu64 " sym=%d load=%.2fs\n", graph_path.c_str(),
                    g.num_vertices, g.num_edges, g.symmetrized ? 1 : 0, load_s);

        TcAblationResult s = run_tc_ablation(g, cfg);
        s.load_s           = load_s;

        std::printf(
            "RESULT graph=%s cull=%s batch=%" PRIu64
            " reps=%d avg_ms=%.3f min_ms=%.3f max_ms=%.3f triangles=%" PRIu64
            " preorient=%d prepass_ms=%.3f oriented_edges=%" PRIu64
            " validated=%d valid=%d cpu_triangles=%" PRIu64 "\n",
            graph_path.c_str(), cfg.cull.c_str(),
            cfg.batch, cfg.reps, s.avg_ms, s.min_ms, s.max_ms, s.triangles,
            cfg.preorient ? 1 : 0, s.prepass_ms, s.oriented_edges, s.validated ? 1 : 0,
            s.valid ? 1 : 0, s.cpu_triangles);
        if (s.validated && !s.valid)
            std::printf("VALIDATION FAILED: gpu=%" PRIu64 " cpu=%" PRIu64 "\n", s.triangles,
                        s.cpu_triangles);

        if (!csv_path.empty()) {
            const bool write_header =
                !std::filesystem::exists(csv_path) || std::filesystem::file_size(csv_path) == 0;
            std::ofstream f(csv_path, std::ios::app);
            if (write_header)
                f << "graph,V,E,cull,batch,reps,warmup,avg_ms,min_ms,"
                     "max_ms,triangles,preorient,prepass_ms,oriented_edges,validated,valid,"
                     "cpu_triangles\n";
            f << graph_path << ',' << g.num_vertices << ',' << g.num_edges
              << ',' << cfg.cull << ',' << cfg.batch
              << ',' << cfg.reps << ',' << cfg.warmup << ',' << s.avg_ms << ',' << s.min_ms << ','
              << s.max_ms << ',' << s.triangles << ',' << (cfg.preorient ? 1 : 0) << ','
              << s.prepass_ms << ',' << s.oriented_edges << ',' << (s.validated ? 1 : 0) << ','
              << (s.valid ? 1 : 0) << ',' << s.cpu_triangles << '\n';
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
