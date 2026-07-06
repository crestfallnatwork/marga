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

# Configuration
GRAPHS = [
    "com-Orkut",
    # Add other graphs below:
    # "com-Friendster",
    # "kron_g500-logn21",
    # "soc-LiveJournal1",
    # "soc-twitter-2010",
    # "GAP-kron",
    # "GAP-road",
    # "GAP-twitter",
    # "GAP-urand",
    # "GAP-web",
]

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
        
    print(f"[{graph_name}] vertices.txt not found. Generating 64 random vertices...")
    
    if not os.path.exists(mtx_file):
        raise FileNotFoundError(f"MTX file not found: {mtx_file}")
        
    num_vertices = 0
    with open(mtx_file, 'r') as f:
        for line in f:
            if line.startswith('%'):
                continue
            # The first non-comment line contains: rows cols nnz
            parts = line.strip().split()
            if len(parts) >= 2:
                num_vertices = int(parts[0])
                break
                
    if num_vertices == 0:
        raise ValueError(f"Could not parse number of vertices from {mtx_file}")
        
    # Generate 64 random vertices
    random.seed(42) # For reproducibility
    sources = [random.randint(0, num_vertices - 1) for _ in range(64)]
    
    with open(vertices_file, 'w') as f:
        for src in sources:
            f.write(f"{src}\n")
            
    print(f"[{graph_name}] Successfully generated vertices.txt (max vertex id: {num_vertices-1})")
    return vertices_file

def run_command(cmd):
    """Runs a shell command and returns stdout."""
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        print(f"Error running command: {' '.join(cmd)}")
        print(result.stdout)
    return result.stdout

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
        pr_out = run_command(pr_cmd)
        results[graph]['PR'] = parse_pr(pr_out)
        
        # 2. Triangle Counting
        print(f"Running TC ({NUM_RUNS} runs)...")
        tc_cmd = [os.path.join(BUILD_DIR, "tc"), mtx_path, "-n", str(NUM_RUNS)]
        tc_out = run_command(tc_cmd)
        results[graph]['TC'] = parse_tc(tc_out)
        
        # 3. BFS
        print(f"Running BFS (64 sources)...")
        bfs_cmd = [os.path.join(BUILD_DIR, "bfs"), mtx_path, "-f", vertices_file]
        bfs_out = run_command(bfs_cmd)
        results[graph]['BFS'] = parse_bfs(bfs_out)
        
        print(f"Results for {graph}: PR={results[graph]['PR']}ms, TC={results[graph]['TC']}ms, BFS={results[graph]['BFS']}ms")

    # Print Markdown Table
    print("\n\n### Benchmark Results")
    print("| Graph | PageRank (ms) | TC (ms) | BFS (ms) |")
    print("|---|---|---|---|")
    for graph, res in results.items():
        pr_val = f"{res['PR']:.2f}" if res['PR'] is not None else "N/A"
        tc_val = f"{res['TC']:.2f}" if res['TC'] is not None else "N/A"
        bfs_val = f"{res['BFS']:.2f}" if res['BFS'] is not None else "N/A"
        print(f"| {graph} | {pr_val} | {tc_val} | {bfs_val} |")

if __name__ == "__main__":
    main()
