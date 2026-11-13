#!/usr/bin/env python3
"""Run every benchmark suite and write the per-algorithm CSV series.

Each system has its own runner module (`run_marga`, `run_cugraph`,
`run_gunrock`, `run_graphrtx`); the selection flags are forwarded unchanged.
A system whose runner is unavailable or raises a clean error is reported and
skipped, so a partial setup still produces the CSVs it can.

    python3 benchmarks/run_all.py
    python3 benchmarks/run_all.py --systems marga,gunrock --graphs GAP-road
    python3 benchmarks/run_all.py --algos tc,bfs --platform a100
"""

import argparse
import importlib
import sys

try:
    from . import config
except ImportError:  # run as a script
    import config


def _runner(system):
    """Import `run_<system>` from this directory or from the package."""
    name = "run_" + system
    try:
        if __package__:
            return importlib.import_module("." + name, __package__)
        return importlib.import_module(name)
    except ImportError as exc:
        raise RuntimeError("runner %s is not available (%s)" % (name, exc))


def _selection(args):
    return argparse.Namespace(graphs=args.graphs, algos=args.algos,
                              platform=args.platform, runs=args.runs,
                              sources=args.sources, out=args.out,
                              timeout=args.timeout)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--systems", default="marga,cugraph,gunrock,graphrtx",
                        help="comma-separated systems (default: %s)"
                             % "marga, cugraph, gunrock, graphrtx")
    parser.add_argument("--graphs", default=",".join(g["name"] for g in config.GRAPHS),
                        help="comma-separated graph names (default: every configured graph)")
    parser.add_argument("--algos", default=",".join(config.ALGORITHMS),
                        help="comma-separated algorithms (default: %s)"
                             % ", ".join(config.ALGORITHMS))
    parser.add_argument("--platform", default="4090",
                        help="platform label used in the CSV file names")
    parser.add_argument("--runs", type=int, default=config.NUM_RUNS,
                        help="timed runs per experiment")
    parser.add_argument("--sources", type=int, default=config.SOURCES,
                        help="sampled sources for BFS and BC")
    parser.add_argument("--out", default=config.OUT_DIR,
                        help="output directory for the CSVs")
    parser.add_argument("--timeout", type=float, default=config.TIMEOUT_S,
                        help="per-invocation timeout in seconds")
    return parser


def main():
    args = _parser().parse_args()
    systems = [system.strip() for system in args.systems.split(",") if system.strip()]
    written = []

    for system in systems:
        print("\n=== %s ===" % system)
        try:
            module = _runner(system)
            paths = module.run(_selection(args))
        except (FileNotFoundError, RuntimeError) as exc:
            print("%s: skipped (%s)" % (system, exc), file=sys.stderr)
            continue
        written.extend(paths or [])

    print("\n%d CSV file(s) written:" % len(written))
    for path in written:
        print("  " + path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
