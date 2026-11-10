#!/usr/bin/env python3
"""Plot peak device memory for one algorithm from the 4090 memory CSV.

Usage:
    python artifacts/plot_memory.py CSV_DIR OUT_DIR [--algorithm tc]

The CSV is ``4090-memory.csv``; ``delta_mib`` is peak minus idle.  The style
matches the benchmark charts in ``plot_graphs.py``: 15x7 in, 24 pt text, log
y-axis and the legend above the axes.  OOM cells are drawn as a small marker
and a rotated label in the series colour.
"""

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator

SERIES = [
    ("cugraph", "cuGraph", "#1f77b4"),
    ("gunrock", "Gunrock", "#ff7f0e"),
    ("graphrtx", "GraphRTX", "#2ca02c"),
    ("marga", "MARGA", "#9467bd"),
]
DATASETS = [
    ("com-Orkut", "com-Orkut"),
    ("soc-twitter-2010", "soc-twitter"),
    ("GAP-twitter", "GAP-twitter"),
    ("indochina-2004", "indochina"),
    ("it-2004", "it-2004"),
]

STYLE = {
    "font.size": 24,
    "axes.labelsize": 30,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
    "legend.fontsize": 24,
    "axes.linewidth": 1.2,
}

YMIN = 0.5
YMAX = 32.0


def load(path, algorithm):
    rows = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["algorithm"] == algorithm:
                rows[(row["system"], row["graph"])] = row
    return rows


def draw(rows, algorithm, out_path):
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(15, 7))

    ticks = [0.5, 1, 2, 4, 8, 16, 32]
    ax.set_yscale("log")
    ax.set_ylim(YMIN, YMAX)
    ax.set_yticks(ticks)
    ax.set_yticklabels(["0.5", "1", "2", "4", "8", "16", "32"])
    ax.yaxis.set_minor_locator(LogLocator(base=2.0, subs=(1.5,), numticks=100))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.grid(axis="y", which="major", color="0.85", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

    x = np.arange(len(DATASETS))
    width = 0.8 / len(SERIES)
    handles = []

    for i, (system, label, colour) in enumerate(SERIES):
        xs = x - 0.4 + width * (i + 0.5)
        heights = []
        for graph, _ in DATASETS:
            row = rows.get((system, graph))
            ok = row is not None and row["status"] == "ok"
            heights.append(int(row["delta_mib"]) / 1024.0 if ok else 0.0)
        bars = ax.bar(
            xs, heights, width * 0.92, color=colour, label=label, zorder=3
        )
        handles.append(bars)
        for xpos, height in zip(xs, heights):
            if height > 0:
                continue
            ax.text(
                xpos,
                YMIN * 1.08,
                "OOM",
                rotation=90,
                ha="center",
                va="bottom",
                fontsize=18,
                color=colour,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in DATASETS])
    ax.set_xlabel("Dataset", labelpad=6)
    ax.set_ylabel("Peak memory (GiB, log scale)")

    fig.legend(
        handles=handles,
        labels=[label for _, label, _ in SERIES],
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
    parser.add_argument("csv_dir")
    parser.add_argument("out_dir")
    parser.add_argument("--algorithm", default="tc")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = load(os.path.join(args.csv_dir, "4090-memory.csv"), args.algorithm)
    stem = "4090_memory_%s" % args.algorithm
    draw(rows, args.algorithm, os.path.join(args.out_dir, stem + ".png"))
    draw(rows, args.algorithm, os.path.join(args.out_dir, stem + "_col.png"))


if __name__ == "__main__":
    main()
