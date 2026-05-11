#!/usr/bin/env python3
"""
Plot speedup of hint and pipeline variants vs baseline from run_benchmarks.sh CSV.

Usage:
  python3 plot_timings.py results/timings_sm80_N4096.csv
  python3 plot_timings.py results/timings_sm80_N4096.csv --out results/speedup.png
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("csv", help="CSV from run_benchmarks.sh")
parser.add_argument("--out", default=None, help="output PNG (default: next to CSV)")
args = parser.parse_args()

csv_path = Path(args.csv)
if not csv_path.exists():
    sys.exit(f"File not found: {csv_path}")

out_path = Path(args.out) if args.out else csv_path.with_suffix(".png")

# ── load data ─────────────────────────────────────────────────────────────────
# data[kernel][variant] = avg_ms
data = {}
with csv_path.open(newline="") as f:
    for row in csv.DictReader(f):
        k, v, ms = row["kernel"], row["variant"], float(row["avg_ms"])
        data.setdefault(k, {})[v] = ms

if not data:
    sys.exit("No data found in CSV.")

kernels = sorted(data.keys())
VARIANTS = ["base", "hint", "pipeline"]
COLORS   = {"base": "#4477AA", "hint": "#EE6677", "pipeline": "#228833"}

# ── print summary table ───────────────────────────────────────────────────────
print(f"\n{'Kernel':<20} {'base (ms)':>10} {'hint (ms)':>10} {'hint Δ':>8} {'pipeline (ms)':>14} {'pipe Δ':>8}")
print("-" * 74)
for k in kernels:
    d    = data[k]
    bv   = d.get("base",     0)
    hv   = d.get("hint",     0)
    pv   = d.get("pipeline", 0)
    hd   = f"{(hv - bv) / bv * 100:+.1f}%" if bv and hv else "n/a"
    pd   = f"{(pv - bv) / bv * 100:+.1f}%" if bv and pv else "n/a"
    hcol = "\033[92m" if hv and hv < bv * 0.99 else ("\033[91m" if hv and hv > bv * 1.01 else "")
    pcol = "\033[92m" if pv and pv < bv * 0.99 else ("\033[91m" if pv and pv > bv * 1.01 else "")
    rst  = "\033[0m"
    print(f"{k:<20} {bv:>10.3f} {hv:>10.3f} {hcol}{hd:>8}{rst} {pv:>14.3f} {pcol}{pd:>8}{rst}")

# ── plot (normalized to baseline) ────────────────────────────────────────────
x  = np.arange(len(kernels))
w  = 0.25
offsets = {"base": -w, "hint": 0, "pipeline": w}

fig, ax = plt.subplots(figsize=(max(10, len(kernels) * 1.4), 6))

for variant, offset in offsets.items():
    vals = []
    for k in kernels:
        bv = data[k].get("base", 0)
        vv = data[k].get(variant, 0)
        vals.append(vv / bv if bv and vv else 0)

    bars = ax.bar(x + offset, vals, w, label=variant,
                  color=COLORS[variant], edgecolor="white", linewidth=0.5)

    if variant != "base":
        for xi, v in enumerate(vals):
            if v > 0:
                pct = (v - 1.0) * 100
                color = "green" if pct < -1 else ("red" if pct > 1 else "gray")
                ax.text(xi + offset, v + 0.01, f"{pct:+.1f}%",
                        ha="center", va="bottom", fontsize=7,
                        color=color, rotation=90)

ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
ax.set_xticks(x)
ax.set_xticklabels(kernels, rotation=20, ha="right", fontsize=9)
ax.set_ylabel("Time normalized to baseline  (lower = faster)")
ax.set_title("Prefetch pass: base vs L2-hint vs software-pipeline\n"
             f"(normalized to base — {csv_path.name})")
ax.legend(handles=[mpatches.Patch(color=COLORS[v], label=v) for v in VARIANTS], fontsize=9)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out_path}")
