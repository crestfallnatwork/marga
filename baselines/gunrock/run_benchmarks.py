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
    # "soc-LiveJournal1",
    # "hollywood-2009",
    # "com-Orkut",
    # "soc-twitter-2010",
    # "GAP-twitter",
    # "com-Friendster",
]

SYNTHETIC = [
    # "delaunay_n24",
    # "kron_g500-logn21",
    # "GAP-kron",
    # "GAP-urand",
]

WEB_CRAWLS = [
    # "indochina-2004",
    # "uk-2005",
    # "webbase-2001",
    # "it-2004",
    # "GAP-web",
]

MISC = [
    # "kmer_V1r",
    # "mawi_201512020330",
    # "nlpkkt240",
]

GRAPHS = ROAD_NETWORKS + SOCIAL_NETWORKS + WEB_CRAWLS + SYNTHETIC + MISC

NUM_RUNS = 5
DATASETS_DIR = "../../datasets"
BUILD_DIR = "build/bin"

def get_vertices_for_bfs(graph_name):
    """
    Reads datasets/<graph_name>/vertices.txt, which is 1-indexed.
    Converts to 0-indexed and returns a comma-separated string.
    """
    graph_dir = os.path.join(DATASETS_DIR, graph_name)
    vertices_file = os.path.join(graph_dir, "vertices.txt")
    
    if not os.path.exists(vertices_file):
        print(f"[{graph_name}] vertices.txt not found. Cannot run BFS.")
        return None
        
    with open(vertices_file, "r") as f:
        lines = f.read().splitlines()
        
    # convert 1-indexed to 0-indexed
    sources = []
    for line in lines:
        line = line.strip()
        if line:
            sources.append(str(int(line) - 1))
            
    return ",".join(sources)

def run_command(cmd):
    """Runs a shell command and returns stdout and a status string ('OK', 'OOM', 'TLE')."""
    try:
        # 30 minutes = 1800 seconds
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=1800)
        if result.returncode != 0:
            print(f"Command failed (likely OOM or crash): {' '.join(cmd)}")
            print("Last 500 characters of output:")
            print(result.stdout[-500:])
            return result.stdout, 'OOM'
        return result.stdout, 'OK'
    except subprocess.TimeoutExpired as e:
        print(f"Command timed out after 30 minutes: {' '.join(cmd)}")
        out = e.stdout if e.stdout else ""
        if isinstance(out, bytes):
            out = out.decode('utf-8', errors='replace')
        return str(out), 'TLE'

def parse_time(output):
    # Looking for: GPU Elapsed Time : 59.6661 (ms)
    matches = re.findall(r"GPU Elapsed Time\s*:\s*([\d\.]+)\s*\(ms\)", output)
    if not matches:
        return None
    times = [float(m) for m in matches]
    return sum(times) / len(times)

def main():
    results = {}

    for graph in GRAPHS:
        print(f"\n{'='*50}\nBenchmarking {graph}\n{'='*50}")
        
        mtx_path = os.path.join(DATASETS_DIR, graph, f"{graph}.mtx")
        
        if not os.path.exists(mtx_path):
            print(f"Skipping {graph}: file {mtx_path} not found.")
            continue
            
        sources_str = get_vertices_for_bfs(graph)
        
        results[graph] = {'PR': None, 'TC': None, 'BFS': None}
        
        # 1. PageRank
        print(f"Running PageRank ({NUM_RUNS} runs)...")
        pr_cmd = [os.path.join(BUILD_DIR, "pr"), "-m", mtx_path, "-n", str(NUM_RUNS)]
        pr_out, pr_status = run_command(pr_cmd)
        results[graph]['PR'] = parse_time(pr_out) if pr_status == 'OK' else pr_status
        
        # 2. Triangle Counting
        print(f"Running TC ({NUM_RUNS} runs)...")
        tc_cmd = [os.path.join(BUILD_DIR, "tc"), "-m", mtx_path, "-n", str(NUM_RUNS)]
        tc_out, tc_status = run_command(tc_cmd)
        results[graph]['TC'] = parse_time(tc_out) if tc_status == 'OK' else tc_status
        
        # 3. BFS
        if sources_str:
            num_sources = len(sources_str.split(','))
            print(f"Running BFS ({num_sources} sources)...")
            bfs_cmd = [os.path.join(BUILD_DIR, "bfs"), "-m", mtx_path, "-s", sources_str]
            bfs_out, bfs_status = run_command(bfs_cmd)
            results[graph]['BFS'] = parse_time(bfs_out) if bfs_status == 'OK' else bfs_status
        else:
            results[graph]['BFS'] = "N/A"
        
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
