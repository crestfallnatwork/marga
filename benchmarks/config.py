"""Shared benchmark configuration.

Every path can be overridden with an environment variable so the same tree
runs on a workstation, a server, or a container:

    MARGA_DATA      directory holding <graph>/<graph>.mtx   (default: ./datasets)
    MARGA_OUT       output directory for the CSVs           (default: ./artifacts/csvs)
    MARGA_BUILD     MARGA build directory                   (default: ./build)
    CUGRAPH_BUILD   cuGraph benchmark build                 (default: ./baselines/cugraph/build)
    GUNROCK_BUILD   GunRock build                           (default: ./baselines/gunrock/build)
    GRAPHTX_BIN     GraphRTX benchmark binary               (default: unset)
"""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.environ.get("MARGA_DATA", os.path.join(ROOT, "datasets"))
OUT_DIR = os.environ.get("MARGA_OUT", os.path.join(ROOT, "artifacts", "csvs"))

MARGA_BUILD = os.environ.get("MARGA_BUILD", os.path.join(ROOT, "build"))
CUGRAPH_BUILD = os.environ.get("CUGRAPH_BUILD", os.path.join(ROOT, "baselines", "cugraph", "build"))
GUNROCK_BUILD = os.environ.get("GUNROCK_BUILD", os.path.join(ROOT, "baselines", "gunrock", "build"))
GRAPHTX_BIN = os.environ.get("GRAPHTX_BIN", "")

# Sampled sources for BFS and BC, and the timed runs per experiment. Run 1 is
# the warm-up; the remaining runs are reported as the median.
SOURCES = 64
NUM_RUNS = 6
TIMEOUT_S = 90 * 60

# Graphs used in the evaluation, ascending by arc count. `V` is the vertex
# count after import; `kind` is the MatrixMarket storage (symmetric files store
# each undirected edge once).
GRAPHS = [
    {"name": "com-Orkut", "V": 3072441, "kind": "symmetric"},
    {"name": "soc-twitter-2010", "V": 21297772, "kind": "symmetric"},
    {"name": "GAP-twitter", "V": 61578415, "kind": "general"},
    {"name": "indochina-2004", "V": 7414866, "kind": "general"},
    {"name": "it-2004", "V": 41291594, "kind": "general"},
    {"name": "GAP-road", "V": 23947347, "kind": "symmetric"},
    {"name": "kron_g500-logn21", "V": 2097152, "kind": "symmetric"},
]

ALGORITHMS = ["tc", "bfs", "cc", "kc", "bc", "pr"]

SYSTEMS = ["marga", "cugraph", "gunrock", "graphrtx_hybrid", "graphrtx_optix"]
