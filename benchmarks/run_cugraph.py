#!/usr/bin/env python3
"""Run the cuGraph baseline and write one CSV per algorithm.

Requires the cuGraph benchmark drivers built in `CUGRAPH_BUILD` (default:
`./baselines/cugraph/build`), one binary per algorithm: `tc`, `bfs`, `cc`,
`bc` and `pr`. k-core has no driver here and is skipped. Paths and data come
from the environment described in `benchmarks/config.py` (`CUGRAPH_BUILD`,
`MARGA_DATA`, `MARGA_OUT`). No conda environment is activated: the drivers
must be executable as built, against whatever library path the build expects.

Each driver is one invocation per graph and reports an `Overall average ...`
line; that value in milliseconds is stored in the CSV. A timeout or a non-zero
exit is recorded as `None`, which `common.write_series` writes as `OOM`.

    python3 benchmarks/run_cugraph.py
    python3 benchmarks/run_cugraph.py --graphs GAP-road,com-Orkut --algos tc,bfs
"""

import argparse
import os

try:
    from . import common, config
except ImportError:  # run as a script
    import common
    import config


# Algorithm -> driver binary name in CUGRAPH_BUILD.
_BINARIES = {
    "tc": "tc",
    "bfs": "bfs",
    "cc": "cc",
    "bc": "bc",
    "pr": "pr",
}

# The single `Overall average ...` line each driver prints, as in
# the original cuGraph driver script.
_PATTERNS = {
    "tc": r"Overall average TC time:\s+([\d.]+)\s+ms",
    "bfs": r"Overall average BFS time across all sources:\s+([\d.]+)\s+ms",
    "cc": r"Overall average CC time:\s+([\d.]+)\s+ms",
    "bc": r"Overall average BC time:\s+([\d.]+)\s+ms",
    "pr": r"Overall average PageRank time:\s+([\d.]+)\s+ms",
}


def _names(value, default):
    """Accept a comma-separated string or an already-split sequence."""
    if value is None:
        return list(default)
    if isinstance(value, str):
        return [name.strip() for name in value.split(",") if name.strip()]
    return list(value)


def _binary(algo):
    """Path of the driver implementing `algo`, checked before the sweep starts."""
    path = os.path.join(config.CUGRAPH_BUILD, _BINARIES[algo])
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return path


def _run_algo(algo, graph, mtx, runs, sources, timeout):
    binary = _binary(algo)
    if algo == "bfs":
        # Same source sample as the other systems, one invocation for all of it.
        cmd = [binary, "-f", common.vertices_path(graph, sources), mtx]
    elif algo == "bc":
        cmd = [binary, "-k", str(sources), mtx]
    else:
        # tc, cc and pr take the matrix path only.
        cmd = [binary, mtx]
    ok, output, _ = common.run(cmd, timeout=timeout)
    if not ok:
        return None
    return common.last_number(output, _PATTERNS[algo])


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
                        help="timed runs per experiment (the drivers fix their own count)")
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

    skipped = [algo for algo in algos if algo not in _BINARIES]
    algos = [algo for algo in algos if algo in _BINARIES]
    if skipped:
        print("skipping unsupported algorithm(s): %s" % ", ".join(skipped))
    if not algos:
        raise RuntimeError("no supported algorithms selected (%s)"
                           % ", ".join(sorted(_BINARIES)))

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

    paths = [common.write_series(platform, "cugraph", algo, values[algo], out)
             for algo in algos]

    columns = ["Graph"] + [algo.upper() for algo in algos]
    rows = [[name] + [values[algo][name] if values[algo][name] is not None else "OOM"
                      for algo in algos]
            for name in values[algos[0]]]
    common.print_table("cuGraph %s (ms)" % platform, columns, rows)
    return paths


def main():
    args = _parser().parse_args()
    result = common.main_guard(lambda: run(args))
    return 0 if result is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
