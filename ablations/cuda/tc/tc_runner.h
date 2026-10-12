#pragma once

#include <cstdint>
#include <string>

#include "host_graph.hpp"

namespace marga_abl {

using pictor::graphs::HostGraph;

struct TcAblationConfig {
    // Pruning level: none | orient | bounded (see TcCull).
    std::string cull = "bounded";
    // Edges per launch; MARGA's TC runs 2^24 edges per batch with a host
    // round-trip per batch. 0 = single launch for the whole graph.
    uint64_t batch = 1ull << 24;
    int  reps   = 3;
    int  warmup = 1;
    bool validate  = false;
    bool use_cache = true;
    // Orientation pre-pass knob; the cull level selects the candidate lists.
    bool preorient = false;
};

struct TcAblationResult {
    double   avg_ms     = 0.0;
    double   min_ms     = 0.0;
    double   max_ms     = 0.0;
    uint64_t triangles  = 0; // normalized: bounded counts once, orient /3, none /6
    uint64_t raw_count  = 0; // as accumulated on the device
    uint64_t cpu_triangles = 0;
    bool     validated = false;
    bool     valid     = false;
    double   load_s    = 0.0;
    // One-time orientation preprocessing cost (0 when --preorient is off).
    double   prepass_ms = 0.0;
    uint64_t oriented_edges = 0;
};

TcAblationResult run_tc_ablation(const HostGraph& g, const TcAblationConfig& cfg);

} // namespace marga_abl
