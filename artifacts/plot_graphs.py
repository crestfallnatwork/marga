#!/usr/bin/env python3
"""Plot one benchmark result chart from the per-series CSV artifacts.

Usage:
    python artifacts/plot_graphs.py CSV_DIR PLATFORM ALGORITHM OUT.png

CSV_DIR contains one file per series, named
``<platform>-<system>-<algorithm>.csv`` with ``graph,value`` rows and no
header. Datasets and systems are selected and ordered below. A system without
a file for this algorithm is skipped, as is a dataset without a row. Any
non-numeric value is drawn as an OOM marker; the CSV retains the detailed
status token.
"""

import argparse
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator

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

YLABELS = {"bc": "Time per source (ms, log scale)"}
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
    ymin = 1
    ymax = 10 ** math.ceil(math.log10(max(values) * 4))
    if ymax <= ymin:
        ymax = ymin * 10

    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.set_yscale("log")
    ax.set_ylim(ymin, ymax)
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=100))
    ax.yaxis.set_minor_locator(
        LogLocator(base=10.0, subs=tuple(range(2, 10)), numticks=100)
    )

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
    ax.grid(axis="y", which="major", color="0.85", linewidth=0.7, zorder=0)
    ax.grid(axis="y", which="minor", color="0.93", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_dir", help="directory with the series CSVs")
    parser.add_argument("platform", help="CSV platform prefix, e.g. 4090")
    parser.add_argument("algorithm", help="CSV algorithm suffix, e.g. bfs")
    parser.add_argument("out", help="output image path")
    parser.add_argument("--ylabel", default=None, help="y-axis label")
    args = parser.parse_args()

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    ylabel = args.ylabel or YLABELS.get(args.algorithm, DEFAULT_YLABEL)
    plot(args.csv_dir, args.platform, args.algorithm, args.out, ylabel)


if __name__ == "__main__":
    main()
