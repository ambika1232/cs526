#!/usr/bin/env python3
"""
Plot hint/pipeline % change vs baseline as a function of N.
Reads all report_{kernel}_{variant}_N{N}.nsys-rep files from a reports directory.

Usage: python3 plot_sweep.py [reports_dir] [--out sweep_results.png] [--kernels K1 K2 ...]
"""

import subprocess, os, re, sys, argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("reports_dir", nargs="?", default="reports")
parser.add_argument("--out", default="sweep_results.png")
parser.add_argument("--kernels", nargs="+", default=None,
                    help="subset of kernels to plot (default: all found)")
args = parser.parse_args()

REPORTS_DIR = args.reports_dir
if not os.path.isdir(REPORTS_DIR):
    sys.exit(f"reports dir not found: {REPORTS_DIR}")

# ── parse all reports ────────────────────────────────────────────────────────
# data[benchmark][N][sub_kernel][variant] = avg_ms
data = {}

pat = re.compile(r"report_(.+?)_(base|hint|pipeline)_N(\d+)\.nsys-rep$")

all_files = sorted(os.listdir(REPORTS_DIR))
total = sum(1 for f in all_files if pat.match(f))
done = 0

for fname in all_files:
    m = pat.match(fname)
    if not m:
        continue
    bench, variant, n_str = m.group(1), m.group(2), m.group(3)
    N = int(n_str)
    fpath = os.path.join(REPORTS_DIR, fname)

    done += 1
    print(f"  [{done}/{total}] parsing {fname}", end="\r")

    try:
        out = subprocess.check_output(
            ["nsys", "stats", "--force-export=true", fpath],
            stderr=subprocess.DEVNULL, text=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"\n  WARNING: nsys stats failed for {fname}")
        continue

    in_section = False
    for line in out.splitlines():
        if "CUDA GPU Kernel Summary" in line:
            in_section = True
            continue
        if in_section:
            if "kern_imp" not in line:
                if line.startswith(" **") or line.strip().startswith("**"):
                    in_section = False
                continue
            parts = line.split()
            kname = re.sub(r"^__clang_ocl_kern_imp_", "", parts[-1])
            avg_ns = float(parts[-6])
            (data
             .setdefault(bench, {})
             .setdefault(N, {})
             .setdefault(kname, {}))[variant] = avg_ns / 1e6

print()

if not data:
    sys.exit("No data parsed — check that reports exist and nsys stats works.")

# ── filter to requested kernels ───────────────────────────────────────────────
if args.kernels:
    data = {k: v for k, v in data.items() if k in args.kernels}

benches = sorted(b for b in data.keys() if b != "pipeline_bench")
N_values_global = sorted({N for b in data.values() for N in b.keys()})

VARIANTS = ["hint", "pipeline"]
COLORS   = {"hint": "#EE6677", "pipeline": "#228833"}
MARKERS  = {"hint": "o", "pipeline": "s"}

# ── one subplot per benchmark ─────────────────────────────────────────────────
ncols = 3
nrows = (len(benches) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                         sharex=False, sharey=False)
axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

fig.suptitle(
    "Hint vs Pipeline % change vs baseline  —  sweep over N\n"
    "(negative = faster than baseline)",
    fontsize=13, fontweight="bold"
)

for ax_idx, bench in enumerate(benches):
    ax = axes_flat[ax_idx]
    bench_data = data[bench]

    # collect all sub-kernels for this benchmark
    all_knames = sorted({kn
                         for n_data in bench_data.values()
                         for kn in n_data.keys()})

    def shorten(kname):
        short = kname
        for prefix in [bench + "_", bench]:
            if short.startswith(prefix):
                return short[len(prefix):]
        # prefix didn't match (e.g. bench="2mm", kname="mm2_kernel1")
        # drop the first underscore-separated word
        parts = kname.split("_", 1)
        return parts[1] if len(parts) > 1 else kname

    plotted_anything = False
    for variant in VARIANTS:
        for kname in all_knames:
            xs, ys = [], []
            for N in sorted(bench_data.keys()):
                kdata = bench_data[N].get(kname, {})
                bv = kdata.get("base", 0)
                vv = kdata.get(variant, 0)
                if bv > 0 and vv > 0:
                    xs.append(N)
                    ys.append((vv - bv) / bv * 100)

            # need at least 2 points to draw a meaningful line
            if len(xs) < 2:
                continue

            label = f"{variant} [{shorten(kname)}]"
            ls = "-" if variant == "hint" else "--"
            ax.plot(xs, ys,
                    color=COLORS[variant],
                    marker=MARKERS[variant],
                    linestyle=ls,
                    markersize=4,
                    linewidth=1.5,
                    alpha=0.85,
                    label=label)
            plotted_anything = True

    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("N", fontsize=9)
    ax.set_ylabel("% change vs base", fontsize=9)
    ax.set_title(bench, fontsize=10, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    if plotted_anything:
        ax.legend(fontsize=7, loc="best")

# hide unused subplots
for ax_idx in range(len(benches), len(axes_flat)):
    axes_flat[ax_idx].set_visible(False)

plt.tight_layout()
plt.savefig(args.out, dpi=150, bbox_inches="tight")
print(f"Saved: {args.out}")

# ── summary table ─────────────────────────────────────────────────────────────
print(f"\n{'Benchmark':<20} {'Sub-kernel':<22}", end="")
for N in N_values_global:
    print(f"  {'N='+str(N):>8}", end="")
print()
print("-" * (42 + 10 * len(N_values_global)))

for bench in benches:
    bench_data = data[bench]
    all_knames = sorted({kn for nd in bench_data.values() for kn in nd.keys()})
    for kname in all_knames:
        for variant in VARIANTS:
            short = kname
            for prefix in [bench + "_", bench]:
                if short.startswith(prefix):
                    short = short[len(prefix):]
            row_label = f"{bench}/{variant}"
            print(f"{row_label:<20} {short:<22}", end="")
            for N in N_values_global:
                kdata = bench_data.get(N, {}).get(kname, {})
                bv = kdata.get("base", 0)
                vv = kdata.get(variant, 0)
                if bv > 0 and vv > 0:
                    pct = (vv - bv) / bv * 100
                    color = "\033[92m" if pct < -1 else ("\033[91m" if pct > 1 else "")
                    print(f"  {color}{pct:>+7.1f}%\033[0m", end="")
                else:
                    print(f"  {'n/a':>8}", end="")
            print()
