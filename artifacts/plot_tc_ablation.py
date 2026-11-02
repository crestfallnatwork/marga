#!/usr/bin/env python3
"""TC ablation artifact: static CUDA arms vs MARGA, one bar per arm.

Data: the recorded matrix in `scratch/TC_ABLATION.md` ("TC ablation matrix —
complete", 2026-10-08/09, RTX 4090, same session). `bounded_pre` is end-to-end:
the one-time orientation pre-pass plus the TC run, because MARGA orients on the
fly. Nothing is recomputed here.

Same layout as the benchmark charts (`plot_graphs.py`), different palette: the
ablation isolates hardnesses within one system, so its colours are not the
library colours.

Usage: scratch/.venv/bin/python artifacts/plot_tc_ablation.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "plots")

# Graph order and values from scratch/TC_ABLATION.md (published matrix).
ROWS = [
    ("GAP-road",         2.31,   2.22,   1.805,  3.40,   8.08),
    ("soc-LiveJournal1", 214.1,  79.1,   38.7,   36.07,  34.54),
    ("com-Orkut",        2219.0, 641.4,  293.8,  255.0,  178.63),
    ("hollywood-2009",   3789.0, 2128.0, 573.6,  568.8,  181.46),
    ("soc-twitter-2010", 58129.0, 27799.0, 12887.0, 11898.0, 2835.1),
]

COLS = ["static_none", "static_orient", "static_bounded",
        "static_bounded_pre", "marga"]
NAMES = ["static none", "static orient", "static bounded",
         "static bounded + pre", "MARGA"]
COLORS = ["#9ecae1", "#4292c6", "#08519c", "#CC79A7", "#D55E00"]

df = pd.DataFrame([r[1:] for r in ROWS], index=[r[0] for r in ROWS], columns=COLS)

ax = df.rename(columns=dict(zip(COLS, NAMES))).plot(
    kind="bar", figsize=(15, 7), width=0.8, color=COLORS, logy=True)

ax.set_ylabel("Time (ms)")
ax.set_xlabel("Dataset")
ax.set_title("TC ablation | Target: 4090")
ax.set_ylim(bottom=1)
ax.grid(True, which="major", ls="-", alpha=0.2, axis="y")
ax.grid(True, which="minor", ls="--", alpha=0.1, axis="y")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

os.makedirs(OUT, exist_ok=True)
out_path = os.path.join(OUT, "4090_tc_ablation.png")
plt.savefig(out_path, dpi=300)
plt.close()
print("wrote", out_path)
