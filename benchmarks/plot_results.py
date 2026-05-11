#!/usr/bin/env python3
"""
Plot wall-clock times (avg GPU kernel time) from nsys reports.
Usage: python3 plot_results.py [reports_dir]
       reports_dir defaults to ./reports
"""

import subprocess, os, re, sys, argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── argument parsing ─────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("reports_dir", nargs="?", default="reports",
                    help="directory containing .nsys-rep files")
parser.add_argument("--out", default="results.png",
                    help="output PNG path")
parser.add_argument("--N", type=int, default=4096,
                    help="only include reports with this N value (default: 4096)")
args = parser.parse_args()

REPORTS_DIR = args.reports_dir
if not os.path.isdir(REPORTS_DIR):
    sys.exit(f"reports dir not found: {REPORTS_DIR}")

# ── parse all reports ────────────────────────────────────────────────────────
# data[benchmark][sub_kernel][variant] = avg_ms
data = {}

pat = re.compile(r"report_(.+?)_(base|hint|pipeline)(?:_N(\d+))?\.nsys-rep$")

for fname in sorted(os.listdir(REPORTS_DIR)):
    m = pat.match(fname)
    if not m:
        continue
    bench, variant = m.group(1), m.group(2)
    file_n = int(m.group(3)) if m.group(3) else None
    if args.N is not None and file_n != args.N and bench != "pipeline_bench":
        continue
    fpath = os.path.join(REPORTS_DIR, fname)

    try:
        out = subprocess.check_output(
            ["nsys", "stats", "--force-export=true", fpath],
            stderr=subprocess.DEVNULL, text=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"  WARNING: nsys stats failed for {fname}")
        continue

    in_section = False
    for line in out.splitlines():
        if "CUDA GPU Kernel Summary" in line:
            in_section = True
            continue
        if in_section:
            if "kern_imp" not in line:
                # exit only on a new section header, not blank/separator lines
                if line.startswith(" **") or (line.strip().startswith("**")):
                    in_section = False
                continue
            # columns: Time(%) TotalNs Instances Avg Med Min Max StdDev Name
            parts = line.split()
            kernel_full = parts[-1]
            avg_ns = float(parts[-6])   # index -6 = Avg; -5 = Med
            # strip clang OCL prefix and clean up
            kname = re.sub(r"^__clang_ocl_kern_imp_", "", kernel_full)

            data.setdefault(bench, {}).setdefault(kname, {})[variant] = avg_ns / 1e6

# ── separate fast vs slow kernels for readability ────────────────────────────
VARIANTS = ["base", "hint", "pipeline"]
COLORS   = {"base": "#4477AA", "hint": "#EE6677", "pipeline": "#228833"}

# Build flat list of (bench, subkernel) entries for each group
def build_entries(bench_filter):
    entries = []
    for bench in sorted(data):
        if not bench_filter(bench):
            continue
        for kname in sorted(data[bench]):
            entries.append((bench, kname))
    return entries

def is_slow(b):
    return any(x in b for x in ["gemm", "2mm", "3mm", "syrk", "syr2k"])

def is_fast(b):
    return not is_slow(b)

fast_entries = build_entries(is_fast)
slow_entries = build_entries(is_slow)

def make_labels(entries):
    labels = []
    for bench, kname in entries:
        # shorten label: strip bench prefix from kname if redundant
        short = kname
        for prefix in [bench + "_", bench]:
            if short.startswith(prefix):
                short = short[len(prefix):]
        labels.append(f"{bench}\n{short}" if short != kname else bench)
    return labels

# ── plotting function (normalized to baseline) ────────────────────────────────
def plot_group(ax, entries, title):
    if not entries:
        ax.set_visible(False)
        return

    labels = make_labels(entries)
    n = len(entries)
    x = np.arange(n)
    w = 0.25
    offsets = {"base": -w, "hint": 0, "pipeline": w}

    for variant, offset in offsets.items():
        norm_vals = []
        base_vals = [data.get(b, {}).get(k, {}).get("base", 0) for b, k in entries]
        for bench, kname in entries:
            bv = data.get(bench, {}).get(kname, {}).get("base", 0)
            vv = data.get(bench, {}).get(kname, {}).get(variant, 0)
            norm_vals.append(vv / bv if bv > 0 else 0)

        ax.bar(x + offset, norm_vals, w, label=variant,
               color=COLORS[variant], edgecolor="white", linewidth=0.5)

        # annotate hint/pipeline bars with % change
        if variant != "base":
            for xi, (nv, bv) in enumerate(zip(norm_vals, base_vals)):
                if nv > 0:
                    pct = (nv - 1.0) * 100
                    color = "green" if pct < -1 else ("red" if pct > 1 else "gray")
                    ax.text(xi + offset, nv + 0.008,
                            f"{pct:+.1f}%", ha="center", va="bottom",
                            fontsize=6, color=color, rotation=90)

    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Time normalized to baseline  (1.0 = base, lower = faster)")
    ax.set_title(title)
    ax.legend(handles=[
        mpatches.Patch(color=COLORS[v], label=v) for v in VARIANTS
    ], fontsize=8)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

# ── derive output paths from --out ────────────────────────────────────────────
base_out = args.out.removesuffix(".png")
out_memory  = base_out + "_memory.png"
out_compute = base_out + "_compute.png"

# ── memory-bound plot ─────────────────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(14, 6))
fig1.suptitle("Memory-bound kernels: base vs hint vs pipeline  (N=4096, sm_80)\n"
              "Normalized to baseline — pipeline_bench N=32768",
              fontsize=12, fontweight="bold")
plot_group(ax1, fast_entries,
           "atax, bicg, mvt, gesummv, pipeline_bench")
fig1.tight_layout()
fig1.savefig(out_memory, dpi=150, bbox_inches="tight")
print(f"Saved: {out_memory}")

# ── compute-bound plot ────────────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(14, 6))
fig2.suptitle("Compute-bound kernels: base vs hint vs pipeline  (N=4096, sm_80)\n"
              "Normalized to baseline",
              fontsize=12, fontweight="bold")
plot_group(ax2, slow_entries,
           "gemm, 2mm, 3mm, syrk, syr2k")
fig2.tight_layout()
fig2.savefig(out_compute, dpi=150, bbox_inches="tight")
print(f"Saved: {out_compute}")

# ── print summary table ───────────────────────────────────────────────────────
print(f"\n{'Benchmark':<25} {'Sub-kernel':<22} {'base (ms)':>10} {'hint (ms)':>10} {'hint Δ':>8} {'pipe (ms)':>10} {'pipe Δ':>8}")
print("-" * 95)
for bench, kname in sorted(fast_entries + slow_entries, key=lambda e: (is_slow(e[0]), e)):
    d = data.get(bench, {}).get(kname, {})
    bv = d.get("base", 0)
    hv = d.get("hint", 0)
    pv = d.get("pipeline", 0)
    hd = f"{(hv-bv)/bv*100:+.1f}%" if bv else "n/a"
    pd = f"{(pv-bv)/bv*100:+.1f}%" if bv else "n/a"
    hcolor = "\033[92m" if hv < bv * 0.99 else ("\033[91m" if hv > bv * 1.01 else "")
    pcolor = "\033[92m" if pv < bv * 0.99 else ("\033[91m" if pv > bv * 1.01 else "")
    reset = "\033[0m"
    print(f"{bench:<25} {kname:<22} {bv:>10.3f} {hv:>10.3f} {hcolor}{hd:>8}{reset} {pv:>10.3f} {pcolor}{pd:>8}{reset}")
