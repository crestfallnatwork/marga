#!/usr/bin/env python3
"""Run the MARGA benchmark suite and write one CSV per algorithm.

Each algorithm is one binary invocation per graph; the reported total is the
value stored in the CSV. A timeout or a non-zero exit is recorded as `None`,
which `common.write_series` writes as `OOM`.

    python3 benchmarks/run_marga.py
    python3 benchmarks/run_marga.py --graphs GAP-road,com-Orkut --algos tc,bfs
"""

import argparse
import os

try:
    from . import common, config
except ImportError:  # run as a script
    import common
    import config


def _names(value, default):
    """Accept a comma-separated string or an already-split sequence."""
    if value is None:
        return list(default)
    if isinstance(value, str):
        return [name.strip() for name in value.split(",") if name.strip()]
    return list(value)


def _binary(algo):
    """Path of the binary implementing `algo`, checked before the sweep starts."""
    names = {
        "tc": "tc",
        "bfs": "bfs",
        "pr": "pagerank",
        "cc": "cc",
        "kc": "kc",
        "bc": "bc",
    }
    path = os.path.join(config.MARGA_BUILD, names[algo])
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return path


def run_tc(graph, mtx, runs, sources, timeout):
    cmd = [_binary("tc"), mtx, "-n", str(runs)]
    ok, output, _ = common.run(cmd, timeout=timeout)
    if not ok:
        return None
    return common.last_number(output, r"Average TC Run Total:\s+([\d.]+)\s+ms")


def run_bfs(graph, mtx, runs, sources, timeout):
    vertices = common.vertices_path(graph, sources)
    cmd = [_binary("bfs"), mtx, "-f", vertices]
    ok, output, _ = common.run(cmd, timeout=timeout)
    if not ok:
        return None
    return common.last_number(output, r"AVERAGE TOTAL:\s+([\d.]+)\s+ms")


def run_pr(graph, mtx, runs, sources, timeout):
    cmd = [_binary("pr"), mtx, "-n", str(runs)]
    ok, output, _ = common.run(cmd, timeout=timeout)
    if not ok:
        return None
    return common.last_number(output, r"Average PR Run Total:\s+([\d.]+)\s+ms")


def run_cc(graph, mtx, runs, sources, timeout):
    cmd = [_binary("cc"), mtx, "-n", str(runs), "-s"]
    ok, output, _ = common.run(cmd, timeout=timeout)
    if not ok:
        return None
    return common.last_number(output, r"Average CC total:\s+([\d.]+)\s+ms")


def run_kc(graph, mtx, runs, sources, timeout):
    cmd = [_binary("kc"), mtx, "-n", str(runs)]
    ok, output, _ = common.run(cmd, timeout=timeout)
    if not ok:
        return None
    return common.last_number(output, r"Average KC total:\s+([\d.]+)\s+ms")


def run_bc(graph, mtx, runs, sources, timeout):
    cmd = [_binary("bc"), mtx, "-n", str(runs), "--sources", str(sources)]
    ok, output, _ = common.run(cmd, timeout=timeout)
    if not ok:
        return None
    return common.last_number(output, r"Average BC total:\s+([\d.]+)\s+ms")


def _run_algo(algo, graph, mtx, runs, sources, timeout):
    runners = {
        "tc": run_tc,
        "bfs": run_bfs,
        "pr": run_pr,
        "cc": run_cc,
        "kc": run_kc,
        "bc": run_bc,
    }
    return runners[algo](graph, mtx, runs, sources, timeout)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs", default=",".join(g["name"] for g in config.GRAPHS),
                        help="comma-separated graph names (default: every configured graph)")
    parser.add_argument("--algos", default=",".join(config.ALGORITHMS),
                        help="comma-separated algorithms (default: %s)"
                             % ", ".join(config.ALGORITHMS))
    parser.add_argument("--platform", default="4090",
                        help="platform label used in the CSV file name")
    parser.add_argument("--runs", type=int, default=config.NUM_RUNS,
                        help="timed runs per experiment")
    parser.add_argument("--sources", type=int, default=config.SOURCES,
                        help="sampled sources for BFS and BC")
    parser.add_argument("--out", default=config.OUT_DIR,
                        help="output directory for the CSVs")
    parser.add_argument("--timeout", type=float, default=config.TIMEOUT_S,
                        help="per-invocation timeout in seconds")
    return parser


def run(args):
    """Run the selected algorithms and return the CSV paths written."""
    graphs = _names(getattr(args, "graphs", None),
                    [g["name"] for g in config.GRAPHS])
    algos = _names(getattr(args, "algos", None), config.ALGORITHMS)
    platform = getattr(args, "platform", None) or "4090"
    runs = getattr(args, "runs", None) or config.NUM_RUNS
    sources = getattr(args, "sources", None) or config.SOURCES
    out = getattr(args, "out", None) or config.OUT_DIR
    timeout = getattr(args, "timeout", None) or config.TIMEOUT_S

    known_graphs = [g["name"] for g in config.GRAPHS]
    unknown = [name for name in graphs if name not in known_graphs]
    if unknown:
        raise RuntimeError("unknown graph(s): %s" % ", ".join(unknown))
    unknown = [algo for algo in algos if algo not in config.ALGORITHMS]
    if unknown:
        raise RuntimeError("unknown algorithm(s): %s" % ", ".join(unknown))
    if not graphs:
        raise RuntimeError("no graphs selected")
    if not algos:
        raise RuntimeError("no algorithms selected")

    for algo in algos:
        _binary(algo)

    selected = [g for g in config.GRAPHS if g["name"] in graphs]
    values = {algo: {} for algo in algos}

    for graph in selected:
        name = graph["name"]
        for algo in algos:
            values[algo][name] = None
        mtx = common.mtx_path(name)
        if not os.path.exists(mtx):
            print("skipping %s: missing %s" % (name, mtx))
            continue
        for algo in algos:
            print("running %s on %s..." % (algo, name))
            values[algo][name] = _run_algo(algo, name, mtx, runs, sources, timeout)
            value = values[algo][name]
            print("  %s: %s" % (algo, "OOM" if value is None else "%.2f ms" % value))

    paths = [common.write_series(platform, "pictor", algo, values[algo], out)
             for algo in algos]

    columns = ["Graph"] + [algo.upper() for algo in algos]
    rows = [[name] + [values[algo][name] if values[algo][name] is not None else "OOM"
                      for algo in algos]
            for name in values[algos[0]]]
    common.print_table("MARGA %s (ms)" % platform, columns, rows)
    return paths


def main():
    args = _parser().parse_args()
    result = common.main_guard(lambda: run(args))
    return 0 if result is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
