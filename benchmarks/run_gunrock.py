#!/usr/bin/env python3
"""Run the GunRock baseline and write one CSV per algorithm.

Requires the GunRock example binaries in `GUNROCK_BUILD` (default:
`./baselines/gunrock/build`), either directly as `<dir>/<algo>` or as
`<dir>/bin/<algo>`. Data and output paths come from the environment described
in `benchmarks/config.py` (`GUNROCK_BUILD`, `MARGA_DATA`, `MARGA_OUT`). No
conda environment is activated: the binaries must be executable as built.

Commands, one invocation per (graph, algorithm):

    bfs    <bfs> -m <mtx> -s <comma-separated strided sources>
    pr     <pr> -m <mtx> -n <runs>
    tc     <tc> -m <mtx> -r true
    kcore  <kcore> <mtx>
    bc     <bc> -m <mtx> -k <sources>

BFS and BC run all sources in one process, so one import is amortised. BFS is
reported as `GPU Mean Over Sources` (per-source time); BC as `GPU Elapsed Time`
(the sum over sources, comparable to the other systems' k-source total).

BC needs a binary that accepts `-k <count>`; the stock example takes
`-s <list>` instead. When no `bc` binary exists, BC is recorded as `None`
(`OOM` in the CSV) and noted in the run output. A timeout or a non-zero exit is
also recorded as `None`.

    python3 benchmarks/run_gunrock.py
    python3 benchmarks/run_gunrock.py --graphs GAP-road,com-Orkut --algos tc,bfs
"""

import argparse
import os

try:
    from . import common, config
except ImportError:  # run as a script
    import common
    import config


# Algorithm -> example binary name. BC is optional (see module docstring).
_BINARIES = {
    "tc": "tc",
    "bfs": "bfs",
    "cc": None,  # GunRock has no CC algorithm; rejected below.
    "kc": "kcore",
    "bc": "bc",
    "pr": "pr",
}

# Percent-free patterns: the sum line is printed with and without the "(sum)"
# label depending on the branch.
_PATTERN_SUM = r"GPU Elapsed Time(?:\s*\(sum\))?\s*:\s*([\d.]+)"
_PATTERN_MEAN = r"GPU Mean Over Sources\s*:\s*([\d.]+)"


def _names(value, default):
    """Accept a comma-separated string or an already-split sequence."""
    if value is None:
        return list(default)
    if isinstance(value, str):
        return [name.strip() for name in value.split(",") if name.strip()]
    return list(value)


def _binary(algo, required=True):
    """Path of the binary implementing `algo`, in `<dir>` or `<dir>/bin`."""
    name = _BINARIES[algo]
    if name is None:
        if required:
            raise RuntimeError("GunRock has no %s binary" % algo)
        return None
    candidates = [os.path.join(config.GUNROCK_BUILD, name),
                  os.path.join(config.GUNROCK_BUILD, "bin", name)]
    for path in candidates:
        if os.path.isfile(path):
            return path
    if required:
        raise FileNotFoundError(candidates[0])
    return None


def _vertices(graph, mtx):
    """Vertex count from the MatrixMarket header, with the configured fallback."""
    try:
        with open(mtx) as fh:
            for line in fh:
                if not line.strip() or line.startswith("%"):
                    continue
                return int(line.split()[0])
    except (OSError, ValueError, IndexError):
        pass
    for entry in config.GRAPHS:
        if entry["name"] == graph:
            return entry["V"]
    raise RuntimeError("cannot determine the vertex count of %s" % graph)


def _strided_sources(vertices, count):
    """`count` sources evenly spaced over the vertex range, as a CSV list."""
    count = max(1, count)
    return ",".join(str(i * vertices // count) for i in range(count))


def _sum_ms(output):
    return common.last_number(output, _PATTERN_SUM)


def _mean_ms(output):
    return common.last_number(output, _PATTERN_MEAN)


def run_bfs(graph, mtx, runs, sources, timeout):
    cmd = [_binary("bfs"), "-m", mtx,
           "-s", _strided_sources(_vertices(graph, mtx), sources)]
    ok, output, _ = common.run(cmd, timeout=timeout)
    if not ok:
        return None
    mean = _mean_ms(output)
    return mean if mean is not None else _sum_ms(output)


def run_pr(graph, mtx, runs, sources, timeout):
    cmd = [_binary("pr"), "-m", mtx, "-n", str(runs)]
    ok, output, _ = common.run(cmd, timeout=timeout)
    if not ok:
        return None
    return _sum_ms(output)


def run_tc(graph, mtx, runs, sources, timeout):
    cmd = [_binary("tc"), "-m", mtx, "-r", "true"]
    ok, output, _ = common.run(cmd, timeout=timeout)
    if not ok:
        return None
    return _sum_ms(output)


def run_kc(graph, mtx, runs, sources, timeout):
    cmd = [_binary("kc"), mtx]
    ok, output, _ = common.run(cmd, timeout=timeout)
    if not ok:
        return None
    return _sum_ms(output)


def run_bc(graph, mtx, runs, sources, timeout):
    binary = _binary("bc", required=False)
    if binary is None:
        print("  bc: no GunRock bc binary in %s; recording OOM" % config.GUNROCK_BUILD)
        return None
    cmd = [binary, "-m", mtx, "-k", str(sources)]
    ok, output, _ = common.run(cmd, timeout=timeout)
    if not ok:
        return None
    total = _sum_ms(output)
    return total if total is not None else _mean_ms(output)


def _run_algo(algo, graph, mtx, runs, sources, timeout):
    runners = {
        "tc": run_tc,
        "bfs": run_bfs,
        "kc": run_kc,
        "bc": run_bc,
        "pr": run_pr,
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
                        help="timed runs per experiment (passed to pr)")
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

    skipped = [algo for algo in algos if _BINARIES.get(algo) is None]
    algos = [algo for algo in algos if _BINARIES.get(algo) is not None]
    if skipped:
        print("skipping unsupported algorithm(s): %s" % ", ".join(skipped))
    if not algos:
        raise RuntimeError("no supported algorithms selected (%s)"
                           % ", ".join(sorted(a for a in _BINARIES
                                              if _BINARIES[a] is not None)))

    for algo in algos:
        # BC is optional: a missing bc binary records OOM instead of aborting.
        _binary(algo, required=algo != "bc")

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

    paths = [common.write_series(platform, "gunrock", algo, values[algo], out)
             for algo in algos]

    columns = ["Graph"] + [algo.upper() for algo in algos]
    rows = [[name] + [values[algo][name] if values[algo][name] is not None else "OOM"
                      for algo in algos]
            for name in values[algos[0]]]
    common.print_table("GunRock %s (ms)" % platform, columns, rows)
    return paths


def main():
    args = _parser().parse_args()
    result = common.main_guard(lambda: run(args))
    return 0 if result is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
