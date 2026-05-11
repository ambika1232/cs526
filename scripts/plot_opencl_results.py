#!/usr/bin/env python3
import argparse
import csv
import math
import subprocess
from pathlib import Path


SUPPORTED_KERNELS = [
    ("2mm", 16),
    ("3mm", 16),
    ("atax", 256),
    ("bicg", 256),
    ("bounded_kernels", 256),
    ("coalesced", 256),
    ("convolution_2d", 16),
    ("gemm", 16),
    ("gesummv", 256),
    ("mvt", 256),
    ("offset", 256),
    ("pipeline_bench", 256),
    ("prefetch", 256),
    ("prefetch_loop_invariant", 256),
    ("prefetch_thread", 256),
    ("rejection_tests", 256),
    ("reverse", 256),
    ("strided", 256),
    ("syr2k", 16),
    ("syrk", 16),
]


def parse_bool(value):
    return str(value).strip().lower() in {"1", "true", "yes"}


def read_rows(csv_paths):
    rows = []
    for path in csv_paths:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    row["_source_csv"] = str(path)
                    row["N"] = int(row["N"])
                    row["local_size"] = int(row["local_size"])
                    row["speedup_best"] = float(row["speedup_best"])
                    row["speedup_median"] = float(row["speedup_median"])
                    row["original_median_ms"] = float(row["original_median_ms"])
                    row["optimized_median_ms"] = float(row["optimized_median_ms"])
                    row["correct"] = parse_bool(row["correct"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise RuntimeError(f"Bad row in {path}: {row}") from exc
                rows.append(row)
    return rows


def dedupe_latest_by_benchmark_size(rows):
    latest = {}
    for row in rows:
        key = (row["benchmark"], row["N"])
        source = Path(row["_source_csv"])
        mtime = source.stat().st_mtime if source.exists() else 0.0
        old = latest.get(key)
        if old is None or mtime >= old[0]:
            latest[key] = (mtime, row)
    return [item[1] for item in latest.values()]


def require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return None
    return plt


def group_by(rows, key):
    grouped = {}
    for row in rows:
        grouped.setdefault(row[key], []).append(row)
    return grouped


def geometric_mean(values):
    positive = [v for v in values if v > 0]
    if not positive:
        return 0.0
    return math.exp(sum(math.log(v) for v in positive) / len(positive))


def plot_speedup_lines(rows, out_dir, only_correct):
    plt = require_matplotlib()
    if plt is None:
        plot_speedup_lines_svg(rows, out_dir, only_correct)
        return
    plot_rows = [r for r in rows if r["correct"] or not only_correct]
    by_kernel = group_by(plot_rows, "benchmark")

    fig, ax = plt.subplots(figsize=(12, 7))
    for kernel, kernel_rows in sorted(by_kernel.items()):
        points = sorted(kernel_rows, key=lambda r: r["N"])
        xs = [r["N"] for r in points]
        ys = [r["speedup_median"] for r in points]
        ax.plot(xs, ys, marker="o", linewidth=1.8, label=kernel)

    ax.axhline(1.0, color="black", linewidth=1, linestyle="--")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Problem size N")
    ax.set_ylabel("Median speedup, original / optimized")
    ax.set_title("OpenCL Median Speedup by Kernel and Size")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "speedup_by_kernel.png", dpi=180)
    plt.close(fig)


def plot_speedup_heatmap(rows, out_dir, only_correct):
    plt = require_matplotlib()
    if plt is None:
        plot_speedup_heatmap_svg(rows, out_dir, only_correct)
        return
    plot_rows = [r for r in rows if r["correct"] or not only_correct]
    kernels = sorted({r["benchmark"] for r in plot_rows})
    sizes = sorted({r["N"] for r in plot_rows})
    if not kernels or not sizes:
        return

    index = {(r["benchmark"], r["N"]): r for r in plot_rows}
    data = []
    for kernel in kernels:
        data.append([
            index.get((kernel, size), {}).get("speedup_median", float("nan"))
            for size in sizes
        ])

    finite_values = [
        index.get((kernel, size), {}).get("speedup_median", float("nan"))
        for kernel in kernels
        for size in sizes
    ]
    finite_values = [v for v in finite_values if not math.isnan(v)]
    vmax = max(2.0, max(finite_values) if finite_values else 2.0)

    fig_w = max(8, len(sizes) * 1.25)
    fig_h = max(6, len(kernels) * 0.38)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0.5, vmax=vmax)

    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([str(s) for s in sizes], rotation=45, ha="right")
    ax.set_yticks(range(len(kernels)))
    ax.set_yticklabels(kernels)
    ax.set_title("OpenCL Median Speedup Heatmap")
    ax.set_xlabel("Problem size N")

    for y, kernel in enumerate(kernels):
        for x, size in enumerate(sizes):
            value = data[y][x]
            if math.isnan(value):
                label = "-"
            else:
                label = f"{value:.2f}"
            ax.text(x, y, label, ha="center", va="center", fontsize=7)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Median speedup")
    fig.tight_layout()
    fig.savefig(out_dir / "speedup_heatmap.png", dpi=180)
    plt.close(fig)


def summarize_by_kernel(rows, only_correct):
    plot_rows = [r for r in rows if r["correct"] or not only_correct]
    by_kernel = group_by(plot_rows, "benchmark")
    summary = []
    for kernel, kernel_rows in by_kernel.items():
        best_row = max(kernel_rows, key=lambda r: r["speedup_median"])
        summary.append({
            "benchmark": kernel,
            "geomean": geometric_mean([r["speedup_median"] for r in kernel_rows]),
            "best": best_row["speedup_median"],
            "best_N": best_row["N"],
            "num_points": len(kernel_rows),
        })
    return summary


def plot_summary_bars(rows, out_dir, only_correct):
    plt = require_matplotlib()
    if plt is None:
        return

    summary = summarize_by_kernel(rows, only_correct)
    if not summary:
        return

    for metric, filename, title, xlabel in [
        ("best", "best_speedup_by_kernel.png", "Best Median Speedup by Kernel", "Best median speedup"),
        ("geomean", "geomean_speedup_by_kernel.png", "Geomean Median Speedup by Kernel", "Geomean median speedup"),
    ]:
        ordered = sorted(summary, key=lambda r: r[metric])
        labels = [r["benchmark"] for r in ordered]
        values = [r[metric] for r in ordered]
        colors = ["#2ca02c" if v >= 1.0 else "#d62728" for v in values]

        fig, ax = plt.subplots(figsize=(11, max(6, len(labels) * 0.38)))
        bars = ax.barh(labels, values, color=colors)
        ax.axvline(1.0, color="black", linewidth=1, linestyle="--")
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.grid(True, axis="x", alpha=0.25)
        ax.set_xlim(left=0, right=max(values) * 1.15)

        for bar, row, value in zip(bars, ordered, values):
            suffix = f" @ N={row['best_N']}" if metric == "best" else ""
            ax.text(
                value + max(values) * 0.015,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}x{suffix}",
                va="center",
                fontsize=8,
            )

        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=180)
        plt.close(fig)


def plot_speedup_small_multiples(rows, out_dir, only_correct):
    plt = require_matplotlib()
    if plt is None:
        return

    plot_rows = [r for r in rows if r["correct"] or not only_correct]
    by_size = group_by(plot_rows, "N")
    for size, size_rows in sorted(by_size.items()):
        ordered = sorted(size_rows, key=lambda r: r["speedup_median"])
        labels = [r["benchmark"] for r in ordered]
        values = [r["speedup_median"] for r in ordered]
        colors = ["#2ca02c" if v >= 1.0 else "#d62728" for v in values]

        fig, ax = plt.subplots(figsize=(11, max(6, len(labels) * 0.38)))
        bars = ax.barh(labels, values, color=colors)
        ax.axvline(1.0, color="black", linewidth=1, linestyle="--")
        ax.set_xlabel("Median speedup")
        ax.set_title(f"Median Speedup by Kernel at N={size}")
        ax.grid(True, axis="x", alpha=0.25)
        ax.set_xlim(left=0, right=max(values) * 1.15)

        for bar, value in zip(bars, values):
            ax.text(
                value + max(values) * 0.015,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}x",
                va="center",
                fontsize=8,
            )

        fig.tight_layout()
        fig.savefig(out_dir / f"speedup_bar_N{size}.png", dpi=180)
        plt.close(fig)


def plot_runtime_bars(rows, out_dir, only_correct):
    plt = require_matplotlib()
    if plt is None:
        plot_runtime_bars_svg(rows, out_dir, only_correct)
        return
    plot_rows = [r for r in rows if r["correct"] or not only_correct]
    by_size = group_by(plot_rows, "N")

    for size, size_rows in sorted(by_size.items()):
        size_rows = sorted(size_rows, key=lambda r: r["benchmark"])
        labels = [r["benchmark"] for r in size_rows]
        original = [r["original_median_ms"] for r in size_rows]
        optimized = [r["optimized_median_ms"] for r in size_rows]
        xs = list(range(len(labels)))

        fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.55), 6))
        width = 0.42
        ax.bar([x - width / 2 for x in xs], original, width, label="Original")
        ax.bar([x + width / 2 for x in xs], optimized, width, label="Optimized")
        ax.set_yscale("log")
        ax.set_ylabel("Median runtime, ms")
        ax.set_title(f"OpenCL Median Runtime at N={size}")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"runtime_median_N{size}.png", dpi=180)
        plt.close(fig)


def svg_escape(text):
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def color_for_index(i):
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    return colors[i % len(colors)]


def speedup_fill(value):
    if math.isnan(value):
        return "#eeeeee"
    clamped = max(0.5, min(2.0, value))
    if clamped >= 1.0:
        t = (clamped - 1.0) / 1.0
        r = int(245 * (1 - t) + 36 * t)
        g = int(245 * (1 - t) + 150 * t)
        b = int(245 * (1 - t) + 75 * t)
    else:
        t = (1.0 - clamped) / 0.5
        r = int(245 * (1 - t) + 215 * t)
        g = int(245 * (1 - t) + 48 * t)
        b = int(245 * (1 - t) + 39 * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def write_svg(path, width, height, body):
    path.write_text(
        "\n".join(
            [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                '<rect width="100%" height="100%" fill="white"/>',
                body,
                "</svg>",
            ]
        )
    )


def plot_speedup_lines_svg(rows, out_dir, only_correct):
    plot_rows = [r for r in rows if r["correct"] or not only_correct]
    by_kernel = group_by(plot_rows, "benchmark")
    if not by_kernel:
        return

    sizes = sorted({r["N"] for r in plot_rows})
    y_values = [r["speedup_median"] for r in plot_rows]
    y_max = max(1.1, max(y_values) * 1.08)
    y_min = min(0.0, min(y_values) * 0.95)

    width, height = 1200, 720
    left, right, top, bottom = 90, 260, 60, 90
    plot_w = width - left - right
    plot_h = height - top - bottom

    log_min = math.log2(min(sizes))
    log_max = math.log2(max(sizes))

    def x_pos(size):
        if log_max == log_min:
            return left + plot_w / 2
        return left + (math.log2(size) - log_min) / (log_max - log_min) * plot_w

    def y_pos(value):
        return top + (y_max - value) / (y_max - y_min) * plot_h

    parts = [
        '<style>text{font-family:Arial,sans-serif;font-size:12px}.title{font-size:20px;font-weight:700}.axis{stroke:#222;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}.legend{font-size:11px}</style>',
        f'<text x="{width/2}" y="30" text-anchor="middle" class="title">OpenCL Median Speedup by Kernel and Size</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" class="axis"/>',
        f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" class="axis"/>',
    ]

    for tick in [0.5, 1.0, 2.0, 4.0, 6.0]:
        if y_min <= tick <= y_max:
            y = y_pos(tick)
            parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" class="grid"/>')
            parts.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end">{tick:.1f}x</text>')

    one_y = y_pos(1.0)
    parts.append(f'<line x1="{left}" y1="{one_y:.1f}" x2="{left+plot_w}" y2="{one_y:.1f}" stroke="#111" stroke-dasharray="5 4"/>')

    for size in sizes:
        x = x_pos(size)
        parts.append(f'<text x="{x:.1f}" y="{top+plot_h+24}" text-anchor="middle">{size}</text>')

    for i, (kernel, kernel_rows) in enumerate(sorted(by_kernel.items())):
        color = color_for_index(i)
        points = sorted(kernel_rows, key=lambda r: r["N"])
        coords = [(x_pos(r["N"]), y_pos(r["speedup_median"])) for r in points]
        if len(coords) > 1:
            polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
            parts.append(f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="2"/>')
        for x, y in coords:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{color}"/>')
        ly = top + 20 + i * 20
        lx = left + plot_w + 25
        parts.append(f'<line x1="{lx}" y1="{ly-4}" x2="{lx+18}" y2="{ly-4}" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<text x="{lx+24}" y="{ly}" class="legend">{svg_escape(kernel)}</text>')

    parts.append(f'<text x="{left+plot_w/2}" y="{height-25}" text-anchor="middle">Problem size N</text>')
    parts.append(f'<text x="20" y="{top+plot_h/2}" transform="rotate(-90 20 {top+plot_h/2})" text-anchor="middle">Median speedup</text>')
    write_svg(out_dir / "speedup_by_kernel.svg", width, height, "\n".join(parts))


def plot_speedup_heatmap_svg(rows, out_dir, only_correct):
    plot_rows = [r for r in rows if r["correct"] or not only_correct]
    kernels = sorted({r["benchmark"] for r in plot_rows})
    sizes = sorted({r["N"] for r in plot_rows})
    if not kernels or not sizes:
        return

    index = {(r["benchmark"], r["N"]): r for r in plot_rows}
    cell_w, cell_h = 95, 26
    left, top = 170, 70
    width = left + cell_w * len(sizes) + 40
    height = top + cell_h * len(kernels) + 50
    parts = [
        '<style>text{font-family:Arial,sans-serif;font-size:12px}.title{font-size:20px;font-weight:700}</style>',
        f'<text x="{width/2}" y="30" text-anchor="middle" class="title">OpenCL Median Speedup Heatmap</text>',
    ]
    for x, size in enumerate(sizes):
        parts.append(f'<text x="{left+x*cell_w+cell_w/2}" y="{top-14}" text-anchor="middle">{size}</text>')
    for y, kernel in enumerate(kernels):
        parts.append(f'<text x="{left-8}" y="{top+y*cell_h+17}" text-anchor="end">{svg_escape(kernel)}</text>')
        for x, size in enumerate(sizes):
            row = index.get((kernel, size))
            value = row["speedup_median"] if row else float("nan")
            fill = speedup_fill(value)
            label = "-" if math.isnan(value) else f"{value:.2f}"
            px, py = left + x * cell_w, top + y * cell_h
            parts.append(f'<rect x="{px}" y="{py}" width="{cell_w}" height="{cell_h}" fill="{fill}" stroke="white"/>')
            parts.append(f'<text x="{px+cell_w/2}" y="{py+17}" text-anchor="middle">{label}</text>')
    write_svg(out_dir / "speedup_heatmap.svg", width, height, "\n".join(parts))


def plot_runtime_bars_svg(rows, out_dir, only_correct):
    plot_rows = [r for r in rows if r["correct"] or not only_correct]
    by_size = group_by(plot_rows, "N")
    for size, size_rows in sorted(by_size.items()):
        size_rows = sorted(size_rows, key=lambda r: r["benchmark"])
        labels = [r["benchmark"] for r in size_rows]
        values = [v for r in size_rows for v in (r["original_median_ms"], r["optimized_median_ms"])]
        max_v = max(values) if values else 1.0
        width = max(1000, 80 * len(labels) + 140)
        height = 620
        left, top, bottom = 70, 60, 155
        plot_h = height - top - bottom
        bar_w = 18
        step = (width - left - 40) / max(1, len(labels))

        def y_pos(value):
            return top + plot_h - (value / max_v) * plot_h

        parts = [
            '<style>text{font-family:Arial,sans-serif;font-size:12px}.title{font-size:20px;font-weight:700}</style>',
            f'<text x="{width/2}" y="30" text-anchor="middle" class="title">OpenCL Median Runtime at N={size}</text>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#222"/>',
            f'<line x1="{left}" y1="{top+plot_h}" x2="{width-35}" y2="{top+plot_h}" stroke="#222"/>',
            f'<text x="{width-185}" y="55" fill="#1f77b4">Original</text>',
            f'<text x="{width-185}" y="75" fill="#ff7f0e">Optimized</text>',
        ]
        for i, row in enumerate(size_rows):
            cx = left + i * step + step / 2
            for dx, key, color in [(-bar_w / 2, "original_median_ms", "#1f77b4"), (bar_w / 2, "optimized_median_ms", "#ff7f0e")]:
                value = row[key]
                y = y_pos(value)
                parts.append(f'<rect x="{cx+dx-bar_w/2:.1f}" y="{y:.1f}" width="{bar_w}" height="{top+plot_h-y:.1f}" fill="{color}"/>')
            parts.append(f'<text x="{cx:.1f}" y="{top+plot_h+18}" text-anchor="end" transform="rotate(-45 {cx:.1f} {top+plot_h+18})">{svg_escape(row["benchmark"])}</text>')
        write_svg(out_dir / f"runtime_median_N{size}.svg", width, height, "\n".join(parts))


def write_summary(rows, out_dir, only_correct):
    summary_rows = [r for r in rows if r["correct"] or not only_correct]
    by_kernel = group_by(summary_rows, "benchmark")
    out_path = out_dir / "plot_summary.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "benchmark",
            "num_points",
            "geomean_speedup_median",
            "best_speedup_median",
            "best_N",
            "all_correct",
        ])
        for kernel, kernel_rows in sorted(by_kernel.items()):
            best_row = max(kernel_rows, key=lambda r: r["speedup_median"])
            writer.writerow([
                kernel,
                len(kernel_rows),
                f"{geometric_mean([r['speedup_median'] for r in kernel_rows]):.6f}",
                f"{best_row['speedup_median']:.6f}",
                best_row["N"],
                all(r["correct"] for r in kernel_rows),
            ])


def run_benchmarks(args):
    out_dir = Path(args.input_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    kernels = args.kernels or [kernel for kernel, _ in SUPPORTED_KERNELS]
    local_size_by_kernel = dict(SUPPORTED_KERNELS)

    for kernel in kernels:
        local_size = args.local_size or local_size_by_kernel.get(kernel, 256)
        out_csv = out_dir / f"{kernel}_dims.csv"
        log_path = out_dir / f"{kernel}_dims.log"
        cmd = [
            "python",
            "scripts/opencl.py",
            "--kernel",
            kernel,
            "--sizes",
            *[str(size) for size in args.sizes],
            "--local-size",
            str(local_size),
            "--warmup",
            str(args.warmup),
            "--repeat",
            str(args.repeat),
            "--out",
            str(out_csv),
        ]
        print(f"Running {kernel}: {' '.join(cmd)}")
        with log_path.open("w") as log:
            completed = subprocess.run(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0:
            print(f"  failed: see {log_path}")
        else:
            print(f"  wrote: {out_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot OpenCL benchmark speedups across kernels and sizes."
    )
    parser.add_argument(
        "--input-dir",
        default="bench_results/opencl_all",
        help="Directory containing opencl.py CSV outputs.",
    )
    parser.add_argument(
        "--glob",
        default="*.csv",
        help="CSV glob inside --input-dir.",
    )
    parser.add_argument(
        "--out-dir",
        default="bench_results/plots",
        help="Directory for PNG plots and summary CSV.",
    )
    parser.add_argument(
        "--include-incorrect",
        action="store_true",
        help="Include rows where correctness failed.",
    )
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Plot every CSV row instead of keeping the newest row per benchmark and N.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run scripts/opencl.py across kernels/sizes before plotting.",
    )
    parser.add_argument(
        "--kernels",
        nargs="+",
        default=None,
        help="Subset of kernels to run when --run is set.",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[65536, 262144, 1048576],
        help="Problem sizes to benchmark when --run is set.",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument(
        "--local-size",
        type=int,
        default=None,
        help="Override local size for every kernel when --run is set.",
    )
    args = parser.parse_args()

    if args.run:
        run_benchmarks(args)

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = sorted(input_dir.glob(args.glob))
    if not csv_paths:
        raise SystemExit(f"No CSV files matched {input_dir / args.glob}")

    rows = read_rows(csv_paths)
    if not args.keep_duplicates:
        rows = dedupe_latest_by_benchmark_size(rows)
    only_correct = not args.include_incorrect

    plot_speedup_lines(rows, out_dir, only_correct)
    plot_speedup_heatmap(rows, out_dir, only_correct)
    plot_summary_bars(rows, out_dir, only_correct)
    plot_speedup_small_multiples(rows, out_dir, only_correct)
    plot_runtime_bars(rows, out_dir, only_correct)
    write_summary(rows, out_dir, only_correct)

    print(f"Read {len(rows)} rows from {len(csv_paths)} CSV files.")
    print(f"Wrote plots and summary to {out_dir}")


if __name__ == "__main__":
    main()
