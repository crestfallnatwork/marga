import argparse
import subprocess
import re
import sys
import statistics

def run_pr_benchmark(graph_file, pr_iters=100, damping=0.85, epsilon=1e-4, runs=5, executable='./build/bench_all'):
    print(f"Running PageRank benchmark on {graph_file}")
    print(f"Config: iterations={pr_iters}, damping={damping}, epsilon={epsilon}, repeat_runs={runs}")

    # Regex patterns for PR instrumentation
    # Standard: "    ALGO TOTAL (start to end): 558.33 ms"
    # Hybrid:   "    ALGO TOTAL (start to end): 469.588 ms"
    # We use a multi-step approach to distinguish Standard vs Hybrid sections
    
    optix_times = []
    hybrid_times = []

    for idx in range(runs):
        cmd = [
            executable, graph_file, 
            "--algo", "pr", 
            "--pr-iters", str(pr_iters), 
            "--pr-damp", str(damping),
            "--pr-eps", str(epsilon),
            "--quiet"
        ]
        print(f"  [{idx+1}/{runs}] Executing...", end="", flush=True)

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            output = result.stdout
            
            # Split output into Standard and Hybrid sections to avoid regex collisions
            sections = output.split("<------")
            
            algo_total_regex = re.compile(r"CORE ALGO \(No Sync\):\s+([\d\.]+)\s+ms")
            iter_regex = re.compile(r"Iterations:\s+(\d+)")
            error_regex = re.compile(r"Iter \d+ error:\s+([0-9\.eE\-\+]+)")
            
            current_optix = None
            current_hybrid = None
            optix_iters = None
            hybrid_iters = None
            optix_errors = []
            hybrid_errors = []

            for sec in sections:
                if "RUN PR" in sec and "Hybrid" not in sec:
                    match = algo_total_regex.search(sec)
                    if match:
                        current_optix = float(match.group(1))
                    it_match = iter_regex.search(sec)
                    if it_match:
                        optix_iters = int(it_match.group(1))
                    optix_errors = [float(m) for m in error_regex.findall(sec)]
                elif "RUN PR Hybrid" in sec:
                    match = algo_total_regex.search(sec)
                    if match:
                        current_hybrid = float(match.group(1))
                    it_match = iter_regex.search(sec)
                    if it_match:
                        hybrid_iters = int(it_match.group(1))
                    hybrid_errors = [float(m) for m in error_regex.findall(sec)]

            if current_optix is not None:
                optix_times.append(current_optix)
                print(f" OptiX: {current_optix:.2f}ms (it: {optix_iters})", end="")
            
            if current_hybrid is not None:
                hybrid_times.append(current_hybrid)
                print(f" | Hybrid: {current_hybrid:.2f}ms (it: {hybrid_iters})", end="")

            if optix_errors:
                print(f"\n    OptiX Errors: {optix_errors}", end="")
            if hybrid_errors:
                print(f"\n    Hybrid Errors: {hybrid_errors}", end="")

            
            print()

        except subprocess.CalledProcessError as e:
            print(f" FAILED.")
            print(f"Stderr:\n{e.stderr}")
        except Exception as e:
             print(f" ERROR: {e}")

    print("\n" + "="*45)
    print("PAGERANK BENCHMARK RESULTS (Averages)")
    print("="*45)
    print(f"Graph: {graph_file}")
    print(f"Successful runs: {len(optix_times)} / {runs}")
    
    if optix_times:
        avg_optix = statistics.mean(optix_times)
        stdev_optix = statistics.stdev(optix_times) if len(optix_times) > 1 else 0
        print(f"Standard PR (OptiX) Average : {avg_optix:.4f} ms (stddev: {stdev_optix:.2f})")
    
    if hybrid_times:
        avg_hybrid = statistics.mean(hybrid_times)
        stdev_hybrid = statistics.stdev(hybrid_times) if len(hybrid_times) > 1 else 0
        print(f"Hybrid PR (CUDA) Average    : {avg_hybrid:.4f} ms (stddev: {stdev_hybrid:.2f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PageRank benchmark multiple times and average results.")
    parser.add_argument("graph_file", help="Path to the .mtx graph file")
    parser.add_argument("--iters", type=int, default=100, help="Number of PR iterations")
    parser.add_argument("--damp", type=float, default=0.85, help="Damping factor")
    parser.add_argument("--eps", type=float, default=1e-4, help="Convergence epsilon")
    parser.add_argument("--runs", type=int, default=5, help="Number of times to repeat the benchmark for averaging")
    parser.add_argument("--exe", default="./build/bench_all", help="Path to the benchmark executable")
    
    args = parser.parse_args()
    
    run_pr_benchmark(args.graph_file, args.iters, args.damp, args.eps, args.runs, args.exe)
