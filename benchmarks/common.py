"""Helpers shared by the benchmark runners.

The runners stay thin: build the command for one (graph, algorithm, system),
run it under a timeout, parse the reported time, and hand the value to
`write_series`, which produces the per-series CSV the plotters read.
"""

import os
import re
import statistics
import subprocess
import sys
import time

try:
    from . import config
except ImportError:  # run as a script
    import config


def mtx_path(graph):
    """MatrixMarket file for `graph`, accepting the flat and nested layouts."""
    candidates = [
        os.path.join(config.DATA_DIR, graph, graph + ".mtx"),
        os.path.join(config.DATA_DIR, graph, graph, graph + ".mtx"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def vertices_path(graph, count=config.SOURCES):
    """Source list for BFS/BC, generated once with gen_vertices if missing."""
    path = os.path.join(config.DATA_DIR, graph, "vertices.txt")
    if os.path.exists(path):
        return path
    mtx = mtx_path(graph)
    if not os.path.exists(mtx):
        raise FileNotFoundError(mtx)
    gen = os.path.join(config.MARGA_BUILD, "gen_vertices")
    result = subprocess.run([gen, mtx, "-n", str(count)], stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True, check=True)
    with open(path, "w") as fh:
        fh.write(result.stdout)
    return path


def run(cmd, timeout=config.TIMEOUT_S, log_dir=None, name=None):
    """Run `cmd`, returning (ok, output, wall_seconds).

    A timeout or a non-zero exit is a recorded outcome, not an error: the
    caller decides whether it is an OOM or an unsupported configuration.
    """
    start = time.monotonic()
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, timeout=timeout)
        output, ok = result.stdout, result.returncode == 0
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        ok = False
    wall = time.monotonic() - start
    if log_dir and name:
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, name + ".log"), "w") as fh:
            fh.write(output)
    return ok, output, wall


def last_number(output, pattern):
    """Last float matched by `pattern` in `output`, or None."""
    matches = re.findall(pattern, output)
    return float(matches[-1]) if matches else None


def median(values):
    return statistics.median(values) if values else None


def write_series(platform, system, algorithm, values, out_dir=None):
    """Write `<platform>-<system>-<algorithm>.csv` as `graph,value` lines.

    `values` maps graph name to a float or None; None is written as `OOM`, which
    the plotters render as a missing bar.
    """
    out_dir = out_dir or config.OUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "%s-%s-%s.csv" % (platform, system, algorithm))
    with open(path, "w") as fh:
        for graph in values:
            value = values[graph]
            fh.write("%s,%s\n" % (graph, "OOM" if value is None else "%.6g" % value))
    return path


def print_table(title, columns, rows):
    """Print a markdown table, matching the original benchmark scripts."""
    print("\n### %s\n" % title)
    print("| " + " | ".join(columns) + " |")
    print("|" + "|".join("---" for _ in columns) + "|")
    for row in rows:
        cells = [("%.2f" % v) if isinstance(v, float) else str(v) for v in row]
        print("| " + " | ".join(cells) + " |")


def main_guard(fn):
    """Report a missing binary or data file as a clean message, not a traceback."""
    try:
        return fn()
    except (FileNotFoundError, RuntimeError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 1
