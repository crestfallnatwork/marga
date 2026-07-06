# AI GENERATED FILE
# SPDX-License-Identifier: Unlicense

# This is free and unencumbered software released into the public domain.
#
# Anyone is free to copy, modify, publish, use, compile, sell, or
# distribute this software, either in source code form or as a compiled
# binary, for any purpose, commercial or non-commercial, and by any
# means.
#
# In jurisdictions that recognize copyright laws, the author or authors
# of this software dedicate any and all copyright interest in the
# software to the public domain. We make this dedication for the benefit
# of the public at large and to the detriment of our heirs and
# successors. We intend this dedication to be an overt act of
# relinquishment in perpetuity of all present and future rights to this
# software under copyright law.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
# OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
#
# For more information, please refer to <http://unlicense.org/>


import os
import subprocess
import random
import re

# Edit this list to add/remove graphs from the benchmark suite

ROAD_NETWORKS = [
    "GAP-road",
    # "europe_osm",
]

SOCIAL_NETWORKS = [
    "soc-LiveJournal1",
    # "hollywood-2009",
    # "com-Orkut",
    # "soc-twitter-2010",
    # "GAP-twitter",
    # "com-Friendster",
]

SYNTHETIC = [
    "delaunay_n24",
    # "kron_g500-logn21",
    # "GAP-kron",
    # "GAP-urand",
]

WEB_CRAWLS = [
    "indochina-2004",
    # "uk-2005",
    # "webbase-2001",
    # "it-2004",
    # "GAP-web",
]

MISC = [
    "kmer_V1r",
    # "mawi_201512020330",
    # "nlpkkt240",
]

GRAPHS = ROAD_NETWORKS + SOCIAL_NETWORKS + WEB_CRAWLS + SYNTHETIC + MISC

NUM_RUNS = 5
DATASETS_DIR = "datasets"
BUILD_DIR = "build"

def ensure_vertices_file(graph_name):
    """
    Checks if datasets/<graph_name>/vertices.txt exists.
    If not, it reads the .mtx file to get the number of vertices
    and generates 64 random vertices, saving them to vertices.txt.
    """
    graph_dir = os.path.join(DATASETS_DIR, graph_name)
    vertices_file = os.path.join(graph_dir, "vertices.txt")
    mtx_file = os.path.join(graph_dir, f"{graph_name}.mtx")
    
    if os.path.exists(vertices_file):
        return vertices_file
        
    print(f"[{graph_name}] vertices.txt not found. Using C++ engine to generate 64 valid sources...")
    
    if not os.path.exists(mtx_file):
        raise FileNotFoundError(f"MTX file not found: {mtx_file}")
        
    cmd = [os.path.join(BUILD_DIR, "gen_vertices"), mtx_file, "-n", "64"]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True, check=True)
    
    with open(vertices_file, "w") as f:
        f.write(result.stdout)
            
    print(f"[{graph_name}] Successfully generated vertices.txt")
    return vertices_file

def run_command(cmd):
    """Runs a shell command and returns stdout and a success flag."""
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        print(f"Command failed (likely OOM or crash): {' '.join(cmd)}")
        print("Last 500 characters of output:")
        print(result.stdout[-500:])
        return result.stdout, False
    return result.stdout, True

def parse_pr(output):
    # Looking for: Average PR Run Total: 606.894 ms
    match = re.search(r"Average PR Run Total:\s+([\d\.]+)\s+ms", output)
    return float(match.group(1)) if match else None

def parse_tc(output):
    # Looking for: Average TC Run Total: 1148.77 ms
    match = re.search(r"Average TC Run Total:\s+([\d\.]+)\s+ms", output)
    return float(match.group(1)) if match else None

def parse_bfs(output):
    # Looking for: AVERAGE TOTAL:       3.421 ms
    match = re.search(r"AVERAGE TOTAL:\s+([\d\.]+)\s+ms", output)
    return float(match.group(1)) if match else None

def main():
    results = {}

    for graph in GRAPHS:
        print(f"\n{'='*50}\nBenchmarking {graph}\n{'='*50}")
        
        mtx_path = os.path.join(DATASETS_DIR, graph, f"{graph}.mtx")
        
        if not os.path.exists(mtx_path):
            print(f"Skipping {graph}: file {mtx_path} not found.")
            continue
            
        vertices_file = ensure_vertices_file(graph)
        
        results[graph] = {'PR': None, 'TC': None, 'BFS': None}
        
        # 1. PageRank
        print(f"Running PageRank ({NUM_RUNS} runs)...")
        pr_cmd = [os.path.join(BUILD_DIR, "pagerank"), mtx_path, "-n", str(NUM_RUNS)]
        pr_out, pr_ok = run_command(pr_cmd)
        results[graph]['PR'] = parse_pr(pr_out) if pr_ok else "OOM"
        
        # 2. Triangle Counting
        print(f"Running TC ({NUM_RUNS} runs)...")
        tc_cmd = [os.path.join(BUILD_DIR, "tc"), mtx_path, "-n", str(NUM_RUNS)]
        tc_out, tc_ok = run_command(tc_cmd)
        results[graph]['TC'] = parse_tc(tc_out) if tc_ok else "OOM"
        
        # 3. BFS
        print(f"Running BFS (64 sources)...")
        bfs_cmd = [os.path.join(BUILD_DIR, "bfs"), mtx_path, "-f", vertices_file]
        bfs_out, bfs_ok = run_command(bfs_cmd)
        results[graph]['BFS'] = parse_bfs(bfs_out) if bfs_ok else "OOM"
        
        print(f"Results for {graph}: PR={results[graph]['PR']}ms, TC={results[graph]['TC']}ms, BFS={results[graph]['BFS']}ms")

    # Print Markdown Table
    print("\n\n### Benchmark Results")
    print("| Graph | PageRank (ms) | TC (ms) | BFS (ms) |")
    print("|---|---|---|---|")
    for graph, res in results.items():
        pr_val = f"{res['PR']:.2f}" if isinstance(res['PR'], float) else (str(res['PR']) if res['PR'] else "N/A")
        tc_val = f"{res['TC']:.2f}" if isinstance(res['TC'], float) else (str(res['TC']) if res['TC'] else "N/A")
        bfs_val = f"{res['BFS']:.2f}" if isinstance(res['BFS'], float) else (str(res['BFS']) if res['BFS'] else "N/A")
        print(f"| {graph} | {pr_val} | {tc_val} | {bfs_val} |")

if __name__ == "__main__":
    main()
