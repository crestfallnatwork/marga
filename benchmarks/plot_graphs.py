#!/usr/bin/env python3
"""Plot one benchmark result chart from the per-series CSV artifacts.

Usage:
    python3 benchmarks/plot_graphs.py [CSV_DIR] PLATFORM ALGORITHM [OUT.png]
    python3 benchmarks/plot_graphs.py [CSV_DIR] PLATFORM DATASET [OUT.png] --by-algorithm

CSV_DIR contains one file per series, named
``<platform>-<system>-<algorithm>.csv`` with ``graph,value`` rows and no
header. It defaults to ``config.OUT_DIR``. Datasets and systems are selected
and ordered below. A system without a file for this algorithm is skipped, as
is a dataset without a row. Any non-numeric value is drawn as an OOM marker;
the CSV retains the detailed status token. With ``--by-algorithm``, the third
argument is a dataset and the x-axis is the algorithm list. OUT defaults to
``artifacts/figs/<platform>_<algorithm>.png``.
"""

import argparse
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator

try:
    from . import config
except ImportError:  # run as a script
    import config

FIGS_DIR = os.path.join(config.ROOT, "artifacts", "figs")

# Datasets to show, in order: (CSV graph name, tick label).
DATASETS = [
    ("com-Orkut", "com-Orkut"),
    ("soc-twitter-2010", "soc-twitter"),
    ("GAP-twitter", "GAP-twitter"),
    ("indochina-2004", "indochina"),
    ("it-2004", "it-2004"),
]

# Systems to draw, in order: (CSV system, legend label, colour).
SERIES = [
    ("cugraph", "cuGraph", "#1f77b4"),
    ("gunrock", "Gunrock", "#ff7f0e"),
    ("graphrtx_hybrid", "GraphRTX", "#2ca02c"),
    ("pictor", "MARGA", "#9467bd"),
]

# Speedup labels are drawn for the other systems relative to this one.
REFERENCE = "pictor"

# Algorithms for the single-dataset comparison, in the paper's results order.
ALGORITHMS = [
    ("tc", "TC"),
    ("bfs", "BFS"),
    ("cc", "CC"),
    ("kc", "kCD"),
    ("bc", "BC"),
    ("pr", "PR"),
]

DEFAULT_YLABEL = "Time (ms, log scale)"

# Figure style.
STYLE = {
    "font.size": 24,
    "axes.labelsize": 30,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
    "legend.fontsize": 24,
    "axes.linewidth": 1.2,
}


def load(path):
    """Return numeric values and OOM markers from a two-column CSV."""
    values = {}
    failures = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            graph, value = line.rsplit(",", 1)
            try:
                values[graph] = float(value)
            except ValueError:
                failures[graph] = "OOM"
    return values, failures


def log_axes(ax, values, headroom=4):
    """Set the common log y-axis and return its lower bound."""
    ymin = 1
    ymax = 10 ** math.ceil(math.log10(max(values) * headroom))
    if ymax <= ymin:
        ymax = ymin * 10
    ax.set_yscale("log")
    ax.set_ylim(ymin, ymax)
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=100))
    ax.yaxis.set_minor_locator(
        LogLocator(base=10.0, subs=tuple(range(2, 10)), numticks=100)
    )
    ax.grid(axis="y", which="major", color="0.85", linewidth=0.7, zorder=0)
    ax.grid(axis="y", which="minor", color="0.93", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    return ymin


def plot(csv_dir, platform, algorithm, out_path, ylabel):
    loaded = []
    for system, label, colour in SERIES:
        path = os.path.join(csv_dir, f"{platform}-{system}-{algorithm}.csv")
        if not os.path.exists(path):
            continue
        values, failures = load(path)
        if any(graph in values or graph in failures for graph, _ in DATASETS):
            loaded.append((system, label, colour, values, failures))
    if not loaded:
        raise SystemExit(
            f"no {algorithm} data for {platform} with the selected datasets"
        )

    values = [
        data[graph]
        for _, _, _, data, _ in loaded
        for graph, _ in DATASETS
        if data.get(graph, 0) > 0
    ]
    if not values:
        raise SystemExit(
            f"no positive {algorithm} values for {platform} "
            "with the selected datasets"
        )
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(15, 7))
    ymin = log_axes(ax, values)

    reference = next(
        (data for system, _, _, data, _ in loaded if system == REFERENCE), {}
    )
    x = np.arange(len(DATASETS))
    width = 0.8 / len(loaded)
    handles = []
    for i, (system, label, colour, data, failures) in enumerate(loaded):
        xs = x - 0.4 + width * (i + 0.5)
        heights = [data.get(graph, 0) for graph, _ in DATASETS]
        bars = ax.bar(xs, heights, width, color=colour, zorder=3)
        bars[0].set_label(label)
        handles.append(bars[0])
        for xpos, (graph, _), height in zip(xs, DATASETS, heights):
            if graph not in data:
                if graph in failures:
                    ax.bar(
                        xpos, ymin * 8, width * 0.12, color=colour, zorder=3
                    )
                    ax.text(
                        xpos,
                        ymin * 12,
                        failures[graph],
                        rotation=90,
                        ha="center",
                        va="bottom",
                        fontsize=18,
                        color=colour,
                    )
                continue
            if system == REFERENCE:
                continue
            base = reference.get(graph, 0)
            if height > 0 and base > 0:
                ax.text(
                    xpos,
                    height * 1.12,
                    f"{height / base:.1f}X",
                    rotation=90,
                    ha="center",
                    va="bottom",
                    fontsize=18,
                    color="black",
                )

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in DATASETS])
    ax.set_xlabel("Dataset", labelpad=6)
    ax.set_ylabel(ylabel)
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(handles),
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
        handlelength=1.0,
        columnspacing=1.6,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_by_algorithm(csv_dir, platform, dataset, out_path, ylabel):
    loaded = []
    for system, label, colour in SERIES:
        values = {}
        failures = {}
        for algorithm, _ in ALGORITHMS:
            path = os.path.join(csv_dir, f"{platform}-{system}-{algorithm}.csv")
            if not os.path.exists(path):
                continue
            rows, failed = load(path)
            if dataset in rows:
                values[algorithm] = rows[dataset]
            elif dataset in failed:
                failures[algorithm] = failed[dataset]
        loaded.append((system, label, colour, values, failures))

    finite = [
        values[algorithm]
        for _, _, _, values, _ in loaded
        for algorithm, _ in ALGORITHMS
        if algorithm in values
    ]
    if not finite:
        raise SystemExit(
            f"no {dataset} data for {platform} across the algorithm list"
        )

    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(15, 7))
    ymin = log_axes(ax, finite)

    reference = next(
        (values for system, _, _, values, _ in loaded if system == REFERENCE), {}
    )
    x = np.arange(len(ALGORITHMS))
    width = 0.8 / len(loaded)
    handles = []
    for i, (system, label, colour, values, failures) in enumerate(loaded):
        xs = x - 0.4 + width * (i + 0.5)
        heights = [values.get(algorithm, 0) for algorithm, _ in ALGORITHMS]
        bars = ax.bar(xs, heights, width, color=colour, zorder=3)
        bars[0].set_label(label)
        handles.append(bars[0])
        for xpos, (algorithm, _), height in zip(xs, ALGORITHMS, heights):
            if algorithm not in values:
                text = failures.get(algorithm, "NA")
                marker_colour = colour if algorithm in failures else "0.55"
                ax.bar(
                    xpos,
                    ymin * 8,
                    width * 0.12,
                    color=marker_colour,
                    zorder=3,
                )
                ax.text(
                    xpos,
                    ymin * 12,
                    text,
                    rotation=90,
                    ha="center",
                    va="bottom",
                    fontsize=18,
                    color=marker_colour,
                )
                continue
            if system == REFERENCE:
                continue
            base = reference.get(algorithm, 0)
            if height > 0 and base > 0:
                ax.text(
                    xpos,
                    height * 1.12,
                    f"{height / base:.1f}X",
                    rotation=90,
                    ha="center",
                    va="bottom",
                    fontsize=18,
                    color="black",
                )

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in ALGORITHMS])
    ax.set_xlabel("Algorithm", labelpad=6)
    ax.set_ylabel(ylabel)
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(handles),
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
        handlelength=1.0,
        columnspacing=1.6,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def resolve_targets(values):
    """Resolve the positionals, which may omit the leading CSV directory.

    Three values are ``CSV_DIR PLATFORM ALGORITHM`` when the first one is a
    directory, and ``PLATFORM ALGORITHM OUT`` otherwise.
    """
    if len(values) == 4:
        return values
    if len(values) == 3:
        if os.path.isdir(values[0]):
            return values + [None]
        return [config.OUT_DIR] + values
    if len(values) == 2:
        return [config.OUT_DIR] + values + [None]
    raise SystemExit("expected [CSV_DIR] PLATFORM ALGORITHM [OUT.png]")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "targets",
        nargs="+",
        metavar="[CSV_DIR] PLATFORM ALGORITHM [OUT.png]",
        help="CSV directory (default: config.OUT_DIR), platform prefix, "
        "algorithm (or dataset), output image (default: artifacts/figs/)",
    )
    parser.add_argument("--ylabel", default=None, help="y-axis label")
    parser.add_argument(
        "--by-algorithm",
        action="store_true",
        help="treat the third argument as a dataset and plot algorithms on x",
    )
    args = parser.parse_args()

    csv_dir, platform, algorithm, out = resolve_targets(args.targets)
    out = out or os.path.join(FIGS_DIR, f"{platform}_{algorithm}.png")
    out_dir = os.path.dirname(out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    ylabel = args.ylabel or DEFAULT_YLABEL
    if args.by_algorithm:
        plot_by_algorithm(csv_dir, platform, algorithm, out, ylabel)
    else:
        plot(csv_dir, platform, algorithm, out, ylabel)


if __name__ == "__main__":
    main()
