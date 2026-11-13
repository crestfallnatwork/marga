#!/usr/bin/env python3
"""Run the GraphRTX baseline (OptiX and hybrid arms) and write one CSV each.

Requires the GraphRTX benchmark binary named by `GRAPHTX_BIN`; the variable
must be set (there is no default) and point at the executable, e.g. the
`bench_all` binary built from the GraphRTX tree. Data and output paths come
from the environment described in `benchmarks/config.py` (`MARGA_DATA`,
`MARGA_OUT`). No conda environment is activated: the binary must be executable
as built.

One process per (graph, algorithm) imports the graph once and runs both arms:

    cc   <bin> <mtx> --algo wcc --undirected   -> WCC: / WCC Hybrid:
    bc   <bin> <mtx> --algo bc --srcs <list>   -> BC:  / BC Hybrid:

CC reports one `WCC: <ms>` and one `WCC Hybrid: <ms>` line; BC reports one line
per source, so its value is the sum over the strided source set (one import,
matching the other systems' k-source total). Time excluding import: only the
per-arm elapsed lines are parsed, never the load/UASP/AABB/BVH setup.

A timeout, an allocation failure or a non-zero exit records `None` for both
arms of that run, which `common.write_series` writes as `OOM`.

    python3 benchmarks/run_graphrtx.py
    python3 benchmarks/run_graphrtx.py --graphs kron_g500-logn21 --algos cc
"""

import argparse
import os
import re

try:
    from . import common, config
except ImportError:  # run as a script
    import common
    import config


# One process emits both arms; `_ARMS[algo]` maps a system to its time line.
_SYSTEMS = ["graphrtx_optix", "graphrtx_hybrid"]
_ALGOS = ["cc", "bc"]
_ARMS = {
    "cc": {
        "graphrtx_optix": r"(?m)^\s*WCC:\s+([\d.]+)\s+ms",
        "graphrtx_hybrid": r"(?m)^\s*WCC Hybrid:\s+([\d.]+)\s+ms",
    },
    "bc": {
        "graphrtx_optix": r"(?m)^\s*BC:\s+([\d.]+)\s+ms",
        "graphrtx_hybrid": r"(?m)^\s*BC Hybrid:\s+([\d.]+)\s+ms",
    },
}


def _names(value, default):
    """Accept a comma-separated string or an already-split sequence."""
    if value is None:
        return list(default)
    if isinstance(value, str):
        return [name.strip() for name in value.split(",") if name.strip()]
    return list(value)


def _binary():
    """Path of the GraphRTX benchmark binary, required for every run."""
    path = config.GRAPHTX_BIN
    if not path:
        raise RuntimeError("GRAPHTX_BIN is not set; point it at the GraphRTX "
                           "benchmark binary (bench_all) before running")
    if not os.path.isfile(path):
        raise FileNotFoundError("GRAPHTX_BIN=%s does not exist" % path)
    return path


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


def _last_ms(output, pattern):
    return common.last_number(output, pattern)


def _sum_ms(output, pattern):
    """Sum every match: BC prints one elapsed line per source."""
    matches = [float(value) for value in re.findall(pattern, output)]
    return sum(matches) if matches else None


def _oom(output):
    low = output.lower()
    return ("out of memory" in low or "bad_alloc" in low
            or "cuda_error_out_of_memory" in low
            or "vk_error_out_of_device_memory" in low)


def _run_algo(algo, graph, mtx, runs, sources, timeout):
    """Run one (graph, algo) process; return {system: ms or None}."""
    binary = _binary()
    if algo == "cc":
        cmd = [binary, mtx, "--algo", "wcc", "--undirected"]
        parse = _last_ms
    else:
        srcs = _strided_sources(_vertices(graph, mtx), sources)
        cmd = [binary, mtx, "--algo", "bc", "--srcs", srcs]
        parse = _sum_ms

    ok, output, _ = common.run(cmd, timeout=timeout)
    result = {system: None for system in _SYSTEMS}
    if not ok:
        print("  %s: %s; recording OOM" % (algo, "OOM" if _oom(output) else
                                           "timeout or non-zero exit"))
        return result
    for system in _SYSTEMS:
        result[system] = parse(output, _ARMS[algo][system])
    return result


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs", default=",".join(g["name"] for g in config.GRAPHS),
                        help="comma-separated graph names (default: every configured graph)")
    parser.add_argument("--algos", default=",".join(_ALGOS),
                        help="comma-separated algorithms (default: %s)"
                             % ", ".join(_ALGOS))
    parser.add_argument("--platform", default="4090",
                        help="platform label used in the CSV file names")
    parser.add_argument("--runs", type=int, default=config.NUM_RUNS,
                        help="timed runs per experiment (GraphRTX times one run)")
    parser.add_argument("--sources", type=int, default=config.SOURCES,
                        help="strided sources for BC")
    parser.add_argument("--out", default=config.OUT_DIR,
                        help="output directory for the CSVs")
    parser.add_argument("--timeout", type=float, default=config.TIMEOUT_S,
                        help="per-invocation timeout in seconds")
    return parser


def run(args):
    """Run the selected algorithms and return the CSV paths written."""
    graphs = _names(getattr(args, "graphs", None),
                    [g["name"] for g in config.GRAPHS])
    algos = _names(getattr(args, "algos", None), _ALGOS)
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

    skipped = [algo for algo in algos if algo not in _ALGOS]
    algos = [algo for algo in algos if algo in _ALGOS]
    if skipped:
        print("skipping unsupported algorithm(s): %s" % ", ".join(skipped))
    if not algos:
        raise RuntimeError("no supported algorithms selected (%s)"
                           % ", ".join(_ALGOS))

    _binary()

    selected = [g for g in config.GRAPHS if g["name"] in graphs]
    values = {(system, algo): {} for algo in algos for system in _SYSTEMS}
    for key in values:
        for graph in selected:
            values[key][graph["name"]] = None

    for graph in selected:
        name = graph["name"]
        mtx = common.mtx_path(name)
        if not os.path.exists(mtx):
            print("skipping %s: missing %s" % (name, mtx))
            continue
        for algo in algos:
            print("running %s on %s..." % (algo, name))
            result = _run_algo(algo, name, mtx, runs, sources, timeout)
            for system in _SYSTEMS:
                values[(system, algo)][name] = result[system]
                value = result[system]
                print("  %s %s: %s" % (system, algo,
                                       "OOM" if value is None else "%.2f ms" % value))

    paths = [common.write_series(platform, system, algo, values[(system, algo)], out)
             for algo in algos for system in _SYSTEMS]

    columns = ["Graph"] + ["%s %s" % (algo.upper(), system.rsplit("_", 1)[-1])
                           for algo in algos for system in _SYSTEMS]
    rows = [[name] + [values[(system, algo)][name]
                      if values[(system, algo)][name] is not None else "OOM"
                      for algo in algos for system in _SYSTEMS]
            for name in values[(_SYSTEMS[0], algos[0])]]
    common.print_table("GraphRTX %s (ms)" % platform, columns, rows)
    return paths


def main():
    args = _parser().parse_args()
    result = common.main_guard(lambda: run(args))
    return 0 if result is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
