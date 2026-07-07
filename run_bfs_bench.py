import argparse
import subprocess
import re
import sys
import statistics

def run_benchmark(graph_file, vertex_file, executable='./build/bench_all'):
    try:
        with open(vertex_file, 'r') as f:
            vertices = [line.strip() for line in f if line.strip().isdigit()]
    except FileNotFoundError:
        print(f"Error: Vertex file '{vertex_file}' not found.")
        sys.exit(1)

    if not vertices:
        print(f"Error: No valid vertices found in '{vertex_file}'.")
        sys.exit(1)

    print(f"Running benchmark on {len(vertices)} vertices from {vertex_file}...")

    # Regex patterns to extract ALGO TOTAL from our custom instrumentation
    # Expecting: "    ALGO TOTAL (while loop): 81.6697 ms"
    algo_pattern = re.compile(r"ALGO TOTAL \(while loop\):\s+([\d\.]+)\s+ms")

    optix_times = []
    hybrid_times = []

    for idx, v in enumerate(vertices):
        cmd = [executable, graph_file, "--algo", "bfs", "--src", str(v), "--quiet"]
        print(f"  [{idx+1}/{len(vertices)}] Running vertex {v}...", end="", flush=True)

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            output = result.stdout
            
            # Find all matches for ALGO TOTAL. 
            # We expect the first match to be Standard BFS and the second to be Hybrid BFS.
            matches = algo_pattern.findall(output)
            
            if len(matches) >= 1:
                optix_time = float(matches[0])
                optix_times.append(optix_time)
                print(f" OptiX: {optix_time:.2f}ms", end="")
                
            if len(matches) >= 2:
                hybrid_time = float(matches[1])
                hybrid_times.append(hybrid_time)
                print(f" | Hybrid: {hybrid_time:.2f}ms", end="")
            
            print() # newline

        except subprocess.CalledProcessError as e:
            print(f" FAILED.")
            print(f"Command returned non-zero exit status {e.returncode}.")
            print(f"Stderr:\n{e.stderr}")
        except Exception as e:
             print(f" ERROR: {e}")

    print("\n" + "="*40)
    print("BENCHMARK RESULTS (Algorithmic Loop Time)")
    print("="*40)
    print(f"Graph: {graph_file}")
    print(f"Vertices tested: {len(optix_times)} / {len(vertices)}")
    
    if optix_times:
        avg_optix = statistics.mean(optix_times)
        print(f"Standard BFS (OptiX) Average : {avg_optix:.4f} ms")
    else:
        print("Standard BFS (OptiX) Average : N/A")

    if hybrid_times:
        avg_hybrid = statistics.mean(hybrid_times)
        print(f"Hybrid BFS (CUDA) Average    : {avg_hybrid:.4f} ms")
    else:
        print("Hybrid BFS (CUDA) Average    : N/A")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BFS benchmark over a list of source vertices.")
    parser.add_argument("graph_file", help="Path to the .mtx graph file (e.g., data/graph.mtx)")
    parser.add_argument("vertex_file", help="Path to a text file containing one source vertex ID per line")
    parser.add_argument("--exe", default="./build/bench_all", help="Path to the benchmark executable")
    
    args = parser.parse_args()
    
    run_benchmark(args.graph_file, args.vertex_file, args.exe)
