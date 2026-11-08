#!/usr/bin/env python3
"""Plot the TC ablation figures from the CSV artifacts.

Usage:
    python artifacts/plot_tc_ablation.py CSV_DIR OUT_DIR

CSV_DIR contains ``4090-tc-ablation.csv`` and
``4090-tc-ablation-phases.csv``. GAP-road is excluded from the figures.
"""

import argparse
import csv
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator

EXCLUDED = {"GAP-road"}
DATASET_LABELS = {
    "com-Orkut": "com-Orkut",
    "soc-twitter-2010": "soc-twitter",
    "GAP-twitter": "GAP-twitter",
    "indochina-2004": "indochina",
    "it-2004": "it-2004",
}
COMBINED_LABELS = {
    "com-Orkut": "com-\nOrkut",
    "soc-twitter-2010": "soc-\ntwitter",
    "GAP-twitter": "GAP-\ntwitter",
    "indochina-2004": "indo-\nchina",
    "it-2004": "it-\n2004",
}
PHASE_ORDER = [
    "com-Orkut",
    "soc-twitter-2010",
    "GAP-twitter",
    "indochina-2004",
    "it-2004",
]

STYLE = {
    "font.size": 24,
    "axes.labelsize": 30,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
    "legend.fontsize": 24,
    "axes.linewidth": 1.2,
}

COMBINED_STYLE = {
    "font.size": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "axes.titlesize": 18,
    "axes.linewidth": 1.2,
}

COLOURS = {
    "compute": "#3d3d3d",
    "compute_prune": "#7f7f7f",
    "no_cull": "#c9b8e8",
    "marga": "#9467bd",
    "task": "#b2d8d8",
    "machinery": "#5ab4ac",
    "mesh": "#01665e",
}


def read_rows(path):
    with open(path, newline="") as fh:
        return [row for row in csv.DictReader(fh) if row["graph"] not in EXCLUDED]


def log_axes(ax, values):
    ymin = 1
    ymax = 10 ** math.ceil(math.log10(max(values) * 2))
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


def grouped(rows, out_path, series, labels):
    graphs = [row["graph"] for row in rows]
    values = {key: [float(row[key]) for row in rows] for key, _, _ in series}
    finite = [value for column in values.values() for value in column]

    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(15, 7))
    log_axes(ax, finite)

    x = np.arange(len(graphs))
    width = 0.8 / len(series)
    handles = []
    for i, (key, label, colour) in enumerate(series):
        xs = x - 0.4 + width * (i + 0.5)
        bars = ax.bar(xs, values[key], width, color=colour, zorder=3)
        bars[0].set_label(label)
        handles.append(bars[0])

    for numerator, denominator in labels:
        numerator_key = series[numerator][0]
        denominator_key = series[denominator][0]
        for i, _ in enumerate(graphs):
            top = values[numerator_key][i]
            base = values[denominator_key][i]
            if top > 0 and base > 0:
                xpos = x[i] - 0.4 + width * (numerator + 0.5)
                ax.text(
                    xpos,
                    top * 1.12,
                    f"{top / base:.1f}X",
                    rotation=90,
                    ha="center",
                    va="bottom",
                    fontsize=18,
                    color="black",
                )

    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS.get(graph, graph) for graph in graphs])
    ax.set_xlabel("Dataset", labelpad=6)
    ax.set_ylabel("Time (ms, log scale)")
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


def phases(rows, out_path):
    by_graph = {row["graph"]: row for row in rows}
    graphs = [graph for graph in PHASE_ORDER if graph in by_graph]
    parts = [
        ("task", "Task", COLOURS["task"]),
        ("machinery", "Pipeline Bookkeeping", COLOURS["machinery"]),
        ("mesh", "Mesh", COLOURS["mesh"]),
    ]
    totals = np.array(
        [
            sum(float(by_graph[graph][key]) for key, _, _ in parts)
            for graph in graphs
        ]
    )

    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(15, 7))
    y = np.arange(len(graphs))
    bottom = np.zeros(len(graphs))
    for key, label, colour in parts:
        shares = (
            100.0
            * np.array([float(by_graph[graph][key]) for graph in graphs])
            / totals
        )
        ax.barh(y, shares, 0.62, left=bottom, color=colour, label=label, zorder=3)
        bottom += shares

    for i, graph in enumerate(graphs):
        share = 100.0 * float(by_graph[graph]["mesh"]) / totals[i]
        ax.text(
            100.0 - share / 2.0,
            i,
            f"{share:.0f}%",
            ha="center",
            va="center",
            color="white",
            fontsize=24,
        )

    ax.set_yticks(y)
    ax.set_yticklabels([DATASET_LABELS.get(graph, graph) for graph in graphs])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 50, 100])
    ax.set_xlabel("Share of pipeline time (%)", labelpad=6)
    ax.grid(axis="x", which="major", color="0.85", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles=handles,
        labels=labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
        handlelength=1.0,
        columnspacing=1.6,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def combined(rows, out_path):
    by_graph = {row["graph"]: row for row in rows}
    graphs = [graph for graph in PHASE_ORDER if graph in by_graph]
    panels = [
        (
            "(a) Culling",
            [
                ("marga_none", "MARGA, no cull", COLOURS["no_cull"]),
                ("marga", "MARGA", COLOURS["marga"]),
            ],
        ),
        (
            "(b) Workload Redistribution",
            [
                ("static_none", "compute", COLOURS["compute"]),
                ("marga_none", "MARGA, no cull", COLOURS["no_cull"]),
            ],
        ),
        (
            "(c) MSP Pipeline",
            [
                ("static_bounded_pre", "compute + prune", COLOURS["compute_prune"]),
                ("marga", "MARGA", COLOURS["marga"]),
            ],
        ),
    ]

    plt.rcParams.update(COMBINED_STYLE)
    fig, axes = plt.subplots(1, 3, figsize=(18, 7.0), sharey=True)
    x = np.arange(len(graphs))
    width = 0.38
    handles = {}

    for ax, (title, series) in zip(axes, panels):
        ax.set_yscale("log")
        ax.set_ylim(1, 1e7)
        ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=100))
        ax.yaxis.set_minor_locator(
            LogLocator(base=10.0, subs=tuple(range(2, 10)), numticks=100)
        )
        ax.grid(axis="y", which="major", color="0.85", linewidth=0.7, zorder=0)
        ax.grid(axis="y", which="minor", color="0.93", linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)

        values = [[float(by_graph[graph][key]) for graph in graphs] for key, _, _ in series]
        for i, (key, label, colour) in enumerate(series):
            xs = x - width / 2 + i * width
            bars = ax.bar(xs, values[i], width, color=colour, zorder=3)
            handles.setdefault(label, bars[0])

        for i in range(len(graphs)):
            top, base = values[0][i], values[1][i]
            ax.text(
                x[i] - width / 2,
                top * 1.12,
                f"{top / base:.1f}X",
                rotation=90,
                ha="center",
                va="bottom",
                fontsize=13,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([COMBINED_LABELS.get(graph, graph) for graph in graphs])
        ax.set_title(title, fontweight="bold", pad=10)

    axes[0].set_ylabel("Time (ms, log scale)")
    fig.supxlabel("Dataset", fontsize=18, y=0.02)

    order = ["compute", "compute + prune", "MARGA, no cull", "MARGA"]
    fig.legend(
        [handles[label] for label in order],
        order,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.985),
        handlelength=0.9,
        handletextpad=0.4,
        columnspacing=0.8,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.90])
    fig.subplots_adjust(wspace=0.06)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_dir", help="directory with the ablation CSVs")
    parser.add_argument("out_dir", help="output directory for the figures")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = read_rows(os.path.join(args.csv_dir, "4090-tc-ablation.csv"))
    phase_rows = read_rows(
        os.path.join(args.csv_dir, "4090-tc-ablation-phases.csv")
    )

    grouped(
        rows,
        os.path.join(args.out_dir, "4090_tc_ablation.png"),
        [
            ("static_none", "compute", COLOURS["compute"]),
            ("static_bounded_pre", "compute + prune", COLOURS["compute_prune"]),
            ("marga_none", "MARGA, no cull", COLOURS["no_cull"]),
            ("marga", "MARGA", COLOURS["marga"]),
        ],
        [(0, 3), (1, 3), (2, 3)],
    )
    grouped(
        rows,
        os.path.join(args.out_dir, "4090_tc_ablation_cull.png"),
        [
            ("marga_none", "MARGA, no cull", COLOURS["no_cull"]),
            ("marga", "MARGA", COLOURS["marga"]),
        ],
        [(0, 1)],
    )
    grouped(
        rows,
        os.path.join(args.out_dir, "4090_tc_ablation_redist.png"),
        [
            ("static_none", "compute", COLOURS["compute"]),
            ("marga_none", "MARGA, no cull", COLOURS["no_cull"]),
        ],
        [(0, 1)],
    )
    grouped(
        rows,
        os.path.join(args.out_dir, "4090_tc_ablation_pipeline.png"),
        [
            ("static_bounded_pre", "compute + prune", COLOURS["compute_prune"]),
            ("marga", "MARGA", COLOURS["marga"]),
        ],
        [(0, 1)],
    )
    phases(
        phase_rows,
        os.path.join(args.out_dir, "4090_tc_ablation_phases.png"),
    )
    combined(
        rows,
        os.path.join(args.out_dir, "4090_tc_ablation_combined.png"),
    )


if __name__ == "__main__":
    main()
