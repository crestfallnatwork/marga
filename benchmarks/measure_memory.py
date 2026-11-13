#!/usr/bin/env python3
"""Peak device memory per (system, algorithm, graph).

Runs one process per cell and samples the device memory in use every 20 ms
with ``nvidia-smi``; the row records ``peak - idle`` in MiB, the wall time and
a status of ``ok``, ``oom`` or ``timeout``.

Allocator modes. The timing builds reserve most of the free device memory up
front (MARGA's heap allocator, cuGraph's pool), so they report the reservation
instead of the graph. Every system with a switch is therefore asked to
allocate eagerly:

    marga      MARGA_EAGER=1      allocations go to the VMA allocator
    cugraph    CUGRAPH_NO_POOL=1  cuda_memory_resource instead of a pool
    gunrock    allocates with cudaMalloc directly
    graphrtx   allocates with cudaMalloc directly

Paths come from ``benchmarks/config.py``. Override them in the environment:

    MARGA_DATA       directory holding <graph>/<graph>.mtx   (default: ./datasets)
    MARGA_OUT        output directory for the CSV and logs    (default: ./artifacts/csvs)
    MARGA_BUILD      MARGA build directory                    (default: ./build)
    CUGRAPH_BUILD    cuGraph benchmark build                  (default: ./baselines/cugraph/build)
    GUNROCK_BUILD    GunRock build                            (default: ./baselines/gunrock/build)
    GRAPHTX_BIN      GraphRTX benchmark binary                (default: unset)

``nvidia-smi`` must be on ``PATH``; the sweep drives it directly, through
``stdbuf`` when it is available so that samples stay line buffered.

Usage:
    python3 benchmarks/measure_memory.py --dry-run
    python3 benchmarks/measure_memory.py
    python3 benchmarks/measure_memory.py --systems marga --algos bfs --graphs com-Orkut
    python3 benchmarks/measure_memory.py --out /tmp/memory.csv
"""

import argparse
import csv
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time

try:
    from . import common, config
except ImportError:  # run as a script
    import common
    import config

# Graphs in the sweep order. BC always samples the same source count; the
# batch that fits the device is chosen inside the kernel.
GRAPHS = ["GAP-road", "com-Orkut", "soc-twitter-2010", "indochina-2004",
          "it-2004", "GAP-twitter"]
BC_SOURCES = {graph: config.SOURCES for graph in GRAPHS}
VERTICES = {graph["name"]: graph["V"] for graph in config.GRAPHS}

SYSTEMS = ["marga", "cugraph", "gunrock", "graphrtx"]
ALGOS = ["tc", "bfs", "cc", "kc", "bc", "pr"]

# Cells that do not complete on a 24 GiB device, or are unsupported.
SKIP = {
    ("cugraph", "tc", "GAP-twitter"),
    ("cugraph", "tc", "indochina-2004"),
    ("cugraph", "tc", "it-2004"),
    ("gunrock", "tc", "GAP-twitter"),
    ("gunrock", "tc", "it-2004"),
    ("graphrtx", "tc", "GAP-twitter"),
    ("cugraph", "cc", "GAP-twitter"),
    ("graphrtx", "cc", "GAP-twitter"),
    ("gunrock", "kc", "GAP-twitter"),
    ("gunrock", "kc", "it-2004"),
}
UNSUPPORTED = {("gunrock", "cc"), ("graphrtx", "kc")}

# Allocation mode per system, set for the cell's process only.
EAGER = {
    "marga": {"MARGA_EAGER": "1"},
    "cugraph": {"CUGRAPH_NO_POOL": "1"},
}

SAMPLE = ["nvidia-smi", "--query-gpu=memory.used",
          "--format=csv,noheader,nounits"]
STREAM = (["stdbuf", "-oL"] if shutil.which("stdbuf") else []) + SAMPLE
FIELDS = ["system", "algorithm", "graph", "peak_mib", "idle_mib",
          "delta_mib", "wall_s", "status", "note"]

# The GPU counts as quiet once its usage is within this distance of the lowest
# value seen while waiting, so a cell starts from a settled device.
IDLE_HEADROOM_MIB = 64
IDLE_WAIT_S = 15


def sources(graph):
    """Strided BC sources, one per k-budget slot."""
    count = BC_SOURCES[graph]
    vertices = VERTICES[graph]
    return [str(i * vertices // count) for i in range(count)]


def command(system, algo, graph):
    """Command line for one cell; the dataset path is resolved here."""
    mtx = common.mtx_path(graph)
    count = BC_SOURCES[graph]
    if system == "marga":
        build = config.MARGA_BUILD
        if algo == "bfs":
            return [os.path.join(build, "bfs"), mtx, "-s", "0"]
        if algo == "pr":
            return [os.path.join(build, "pagerank"), mtx, "-n", "1"]
        if algo == "tc":
            return [os.path.join(build, "tc"), mtx, "-n", "1"]
        if algo == "cc":
            argv = [os.path.join(build, "cc"), mtx, "-n", "1"]
            # The paper uses the full arm on GAP-twitter, the symmetric arm
            # on the other graphs.
            return argv if graph == "GAP-twitter" else argv + ["-s"]
        if algo == "kc":
            return [os.path.join(build, "kc"), mtx, "-n", "1"]
        if algo == "bc":
            return [os.path.join(build, "bc"), mtx, "-n", "1", "--sources", str(count)]
    elif system == "cugraph":
        build = config.CUGRAPH_BUILD
        if algo == "bfs":
            return [os.path.join(build, "bfs"), mtx]
        if algo == "pr":
            return [os.path.join(build, "pagerank"), mtx, "-n", "1"]
        if algo == "tc":
            return [os.path.join(build, "tc"), mtx, "-n", "1"]
        if algo == "cc":
            return [os.path.join(build, "cc"), mtx]
        if algo == "kc":
            return [os.path.join(build, "core64"), mtx, "-n", "1"]
        if algo == "bc":
            return [os.path.join(build, "bc"), "-k", str(count), mtx]
    elif system == "gunrock":
        build = os.path.join(config.GUNROCK_BUILD, "bin")
        if algo == "bfs":
            return [os.path.join(build, "bfs"), "-m", mtx, "-s", "0"]
        if algo == "pr":
            return [os.path.join(build, "pr"), "-m", mtx, "-n", "1"]
        if algo == "tc":
            return [os.path.join(build, "tc"), "-m", mtx, "-n", "1"]
        if algo == "kc":
            return [os.path.join(build, "kcore"), mtx]
        if algo == "bc":
            # One process, one import, all sources in one run.
            argv = [os.path.join(build, "bc"), "-m", mtx, "-s"]
            return argv + [",".join(sources(graph))]
    elif system == "graphrtx":
        binary = config.GRAPHTX_BIN
        if not binary:
            raise RuntimeError("GraphRTX needs GRAPHTX_BIN set to its binary")
        if algo == "bfs":
            return [binary, mtx, "--algo", "bfs", "--src", "0"]
        if algo == "pr":
            return [binary, mtx, "--algo", "pr"]
        if algo == "tc":
            return [binary, mtx, "--algo", "tc"]
        if algo == "cc":
            return [binary, mtx, "--algo", "wcc", "--undirected"]
        if algo == "bc":
            argv = [binary, mtx, "--algo", "bc", "--srcs"]
            return argv + [",".join(sources(graph))]
    raise ValueError("unknown cell: %s %s %s" % (system, algo, graph))


def environment(system):
    env = dict(os.environ)
    env.update(EAGER.get(system, {}))
    return env


def preview(cell):
    prefix = ["%s=%s" % item for item in EAGER.get(cell[0], {}).items()]
    return shlex.join(prefix + command(*cell))


def cells(systems, algos, graphs):
    for graph in graphs:
        for system in systems:
            for algo in algos:
                if (system, algo) in UNSUPPORTED:
                    continue
                if (system, algo, graph) in SKIP:
                    continue
                yield system, algo, graph


def gpu_used():
    out = subprocess.check_output(SAMPLE, text=True)
    return int(out.strip().splitlines()[0])


def wait_idle(timeout=IDLE_WAIT_S):
    """Wait for the device to settle and return its usage in MiB."""
    deadline = time.monotonic() + timeout
    floor = gpu_used()
    while time.monotonic() < deadline:
        used = gpu_used()
        floor = min(floor, used)
        if used <= floor + IDLE_HEADROOM_MIB:
            return used
        time.sleep(0.2)
    return gpu_used()


def run_cell(cmd, env, log_path, timeout):
    """Run one cell under the memory sampler; return its measurements."""
    idle = wait_idle()
    sampler = subprocess.Popen(
        STREAM + ["-lms", "20"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, bufsize=1,
    )
    peak = [0]

    def read_samples():
        for line in sampler.stdout:
            try:
                value = int(line.strip())
            except ValueError:
                continue
            if value > peak[0]:
                peak[0] = value

    reader = threading.Thread(target=read_samples, daemon=True)
    reader.start()
    t0 = time.monotonic()
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                env=env)
        try:
            rc = proc.wait(timeout=timeout)
            timed_out = False
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            rc = -9
            timed_out = True
    wall = time.monotonic() - t0
    sampler.terminate()
    try:
        sampler.wait(timeout=2)
    except subprocess.TimeoutExpired:
        sampler.kill()
    reader.join(timeout=2)
    sampler.stdout.close()

    with open(log_path, errors="replace") as log:
        text = log.read().lower()
    if timed_out:
        status = "timeout"
    elif rc == 0:
        status = "ok"
    elif any(token in text for token in ("out of memory", "outofmemory",
                                         "cuda_error_out_of_memory",
                                         "bad_alloc")):
        status = "oom"
    else:
        status = "failed"
    return idle, peak[0], wall, status, rc


def load_done(path):
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("status") == "ok":
                done.add((row["system"], row["algorithm"], row["graph"]))
    return done


def selection(text, allowed, what):
    """Parse a comma-separated subset of `allowed`; empty means all of it."""
    if not text:
        return list(allowed)
    chosen = [item.strip() for item in text.split(",") if item.strip()]
    unknown = [item for item in chosen if item not in allowed]
    if unknown:
        raise SystemExit("unknown %s: %s" % (what, ", ".join(unknown)))
    return chosen


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--systems", default="",
                        help="comma-separated subset of %s" % ", ".join(SYSTEMS))
    parser.add_argument("--algos", default="",
                        help="comma-separated subset of %s" % ", ".join(ALGOS))
    parser.add_argument("--graphs", default="",
                        help="comma-separated subset of %s" % ", ".join(GRAPHS))
    parser.add_argument("--platform", default="4090",
                        help="label in the default CSV name")
    parser.add_argument("--timeout", type=int, default=1800,
                        help="seconds per cell")
    parser.add_argument("--out", default="",
                        help="output CSV (default: <MARGA_OUT>/<platform>-memory.csv)")
    parser.add_argument("--limit", type=int, default=0,
                        help="stop after N cells")
    parser.add_argument("--retry-failed", action="store_true",
                        help="redo cells that are not ok")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the commands and exit")
    args = parser.parse_args()

    todo = list(cells(selection(args.systems, SYSTEMS, "system"),
                      selection(args.algos, ALGOS, "algorithm"),
                      selection(args.graphs, GRAPHS, "graph")))
    if args.limit:
        todo = todo[:args.limit]

    if args.dry_run:
        for cell in todo:
            print(preview(cell))
        print("# %d cells" % len(todo))
        return 0

    out_path = args.out or os.path.join(config.OUT_DIR,
                                        "%s-memory.csv" % args.platform)
    log_dir = os.path.join(os.path.dirname(os.path.abspath(out_path)), "logs")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    done = set() if args.retry_failed else load_done(out_path)
    pending = [cell for cell in todo if cell not in done]
    print("cells: %d selected, %d already done, %d to run"
          % (len(todo), len(todo) - len(pending), len(pending)), flush=True)

    new_file = not os.path.exists(out_path)
    out = open(out_path, "a", newline="")
    writer = csv.DictWriter(out, fieldnames=FIELDS)
    if new_file:
        writer.writeheader()
        out.flush()

    for i, (system, algo, graph) in enumerate(pending, 1):
        log_path = os.path.join(log_dir, "%s-%s-%s.log" % (system, algo, graph))
        cmd = command(system, algo, graph)
        print("[%d/%d] %s %s %s" % (i, len(pending), system, algo, graph),
              flush=True)
        idle, peak, wall, status, rc = run_cell(cmd, environment(system),
                                                log_path, args.timeout)
        delta = max(0, peak - idle)
        writer.writerow({
            "system": system, "algorithm": algo, "graph": graph,
            "peak_mib": peak, "idle_mib": idle, "delta_mib": delta,
            "wall_s": "%.1f" % wall, "status": status,
            "note": "rc=%d" % rc,
        })
        out.flush()
        os.fsync(out.fileno())
        print("    peak=%d idle=%d delta=%d wall=%.1fs %s"
              % (peak, idle, delta, wall, status), flush=True)


if __name__ == "__main__":
    sys.exit(common.main_guard(main))
