#!/usr/bin/env python3
import argparse
import csv
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np

try:
    import pyopencl as cl
except ImportError:
    print("ERROR: pyopencl is not installed.")
    print("Try: python -m pip install pyopencl")
    sys.exit(1)


OPENCL_SRC = r"""
__kernel void strided_naive(
    __global const float *A,
    __global float *B,
    const int N)
{
    int gid = get_global_id(0);
    if (gid < N) {
        B[gid] = A[4 * gid];
    }
}

__kernel void strided_tile_remap(
    __global const float *A,
    __global float *B,
    const int N,
    __local float *tile)
{
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int lsize = get_local_size(0);
    int group = get_group_id(0);

    // Output elements covered by this workgroup start at group*lsize.
    int group_base_out = group * lsize;

    // Original strided load for element gid is A[4*gid].
    // The group needs the dense segment starting at A[4*group_base_out].
    int dense_base = 4 * group_base_out;

    // Need logical indices up to 4*(lsize-1), inclusive.
    int logical_tile_elems = 4 * (lsize - 1) + 1;

    // Cooperative coalesced preload from global A into local tile.
    for (int off = lid; off < logical_tile_elems; off += lsize) {
        int global_idx = dense_base + off;

        // A has length 4*N for this synthetic benchmark.
        if (global_idx < 4 * N) {
            tile[off] = A[global_idx];
        } else {
            tile[off] = 0.0f;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (gid < N) {
        B[gid] = tile[4 * lid];
    }
}

__kernel void coalesced_naive(
    __global const float *A,
    __global float *B,
    const int N)
{
    int gid = get_global_id(0);
    if (gid < N) {
        B[gid] = A[gid];
    }
}

__kernel void coalesced_noop_opt(
    __global const float *A,
    __global float *B,
    const int N)
{
    int gid = get_global_id(0);
    if (gid < N) {
        B[gid] = A[gid];
    }
}
"""


def round_up(x: int, multiple: int) -> int:
    return ((x + multiple - 1) // multiple) * multiple


def event_ms(evt: cl.Event) -> float:
    evt.wait()
    return (evt.profile.end - evt.profile.start) * 1e-6


def best_ms(times):
    return min(times)


def median_ms(times):
    return statistics.median(times)


def mean_ms(times):
    return statistics.mean(times)


def run_kernel(queue, kernel, global_size, local_size, args, warmup, repeat):
    # Warmup
    for _ in range(warmup):
        evt = kernel(queue, (global_size,), (local_size,), *args)
        evt.wait()

    queue.finish()

    times = []
    for _ in range(repeat):
        evt = kernel(queue, (global_size,), (local_size,), *args)
        times.append(event_ms(evt))

    queue.finish()
    return times


def benchmark_strided(ctx, queue, prg, N, local_size, warmup, repeat):
    mf = cl.mem_flags

    # A has 4*N elements because kernel reads A[4*gid].
    A = np.random.rand(4 * N).astype(np.float32)
    B_naive = np.zeros(N, dtype=np.float32)
    B_opt = np.zeros(N, dtype=np.float32)

    A_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    B_naive_buf = cl.Buffer(ctx, mf.WRITE_ONLY, B_naive.nbytes)
    B_opt_buf = cl.Buffer(ctx, mf.WRITE_ONLY, B_opt.nbytes)

    global_size = round_up(N, local_size)
    tile_elems = 4 * (local_size - 1) + 1
    tile_bytes = tile_elems * np.dtype(np.float32).itemsize

    naive_times = run_kernel(
        queue,
        prg.strided_naive,
        global_size,
        local_size,
        [A_buf, B_naive_buf, np.int32(N)],
        warmup,
        repeat,
    )

    opt_times = run_kernel(
        queue,
        prg.strided_tile_remap,
        global_size,
        local_size,
        [A_buf, B_opt_buf, np.int32(N), cl.LocalMemory(tile_bytes)],
        warmup,
        repeat,
    )

    cl.enqueue_copy(queue, B_naive, B_naive_buf).wait()
    cl.enqueue_copy(queue, B_opt, B_opt_buf).wait()

    max_abs_err = float(np.max(np.abs(B_naive - B_opt)))
    ok = bool(np.allclose(B_naive, B_opt, rtol=1e-5, atol=1e-6))

    return {
        "kernel": "strided_tile_remap",
        "N": N,
        "local_size": local_size,
        "global_size": global_size,
        "naive_best_ms": best_ms(naive_times),
        "opt_best_ms": best_ms(opt_times),
        "naive_median_ms": median_ms(naive_times),
        "opt_median_ms": median_ms(opt_times),
        "naive_mean_ms": mean_ms(naive_times),
        "opt_mean_ms": mean_ms(opt_times),
        "speedup_best": best_ms(naive_times) / best_ms(opt_times),
        "speedup_median": median_ms(naive_times) / median_ms(opt_times),
        "correct": ok,
        "max_abs_err": max_abs_err,
    }


def benchmark_coalesced(ctx, queue, prg, N, local_size, warmup, repeat):
    mf = cl.mem_flags

    A = np.random.rand(N).astype(np.float32)
    B_naive = np.zeros(N, dtype=np.float32)
    B_opt = np.zeros(N, dtype=np.float32)

    A_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    B_naive_buf = cl.Buffer(ctx, mf.WRITE_ONLY, B_naive.nbytes)
    B_opt_buf = cl.Buffer(ctx, mf.WRITE_ONLY, B_opt.nbytes)

    global_size = round_up(N, local_size)

    naive_times = run_kernel(
        queue,
        prg.coalesced_naive,
        global_size,
        local_size,
        [A_buf, B_naive_buf, np.int32(N)],
        warmup,
        repeat,
    )

    opt_times = run_kernel(
        queue,
        prg.coalesced_noop_opt,
        global_size,
        local_size,
        [A_buf, B_opt_buf, np.int32(N)],
        warmup,
        repeat,
    )

    cl.enqueue_copy(queue, B_naive, B_naive_buf).wait()
    cl.enqueue_copy(queue, B_opt, B_opt_buf).wait()

    max_abs_err = float(np.max(np.abs(B_naive - B_opt)))
    ok = bool(np.allclose(B_naive, B_opt, rtol=1e-5, atol=1e-6))

    return {
        "kernel": "coalesced_noop",
        "N": N,
        "local_size": local_size,
        "global_size": global_size,
        "naive_best_ms": best_ms(naive_times),
        "opt_best_ms": best_ms(opt_times),
        "naive_median_ms": median_ms(naive_times),
        "opt_median_ms": median_ms(opt_times),
        "naive_mean_ms": mean_ms(naive_times),
        "opt_mean_ms": mean_ms(opt_times),
        "speedup_best": best_ms(naive_times) / best_ms(opt_times),
        "speedup_median": median_ms(naive_times) / median_ms(opt_times),
        "correct": ok,
        "max_abs_err": max_abs_err,
    }


def choose_context(device_substr=None):
    platforms = cl.get_platforms()
    if not platforms:
        raise RuntimeError("No OpenCL platforms found.")

    candidates = []
    for p in platforms:
        for d in p.get_devices():
            candidates.append((p, d))

    if device_substr:
        s = device_substr.lower()
        candidates = [(p, d) for p, d in candidates if s in d.name.lower()]

    if not candidates:
        raise RuntimeError("No matching OpenCL device found.")

    # Prefer GPU, otherwise first available device.
    for p, d in candidates:
        if d.type & cl.device_type.GPU:
            return cl.Context([d]), d

    p, d = candidates[0]
    return cl.Context([d]), d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", type=int,
                    default=[1 << 18, 1 << 20, 1 << 22, 1 << 24])
    ap.add_argument("--local-size", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--repeat", type=int, default=30)
    ap.add_argument("--device", type=str, default=None,
                    help="Substring of OpenCL device name to select.")
    ap.add_argument("--out", type=str, default="bench_results/runtime_speedups.csv")
    args = ap.parse_args()

    ctx, dev = choose_context(args.device)
    queue = cl.CommandQueue(
        ctx,
        properties=cl.command_queue_properties.PROFILING_ENABLE,
    )

    print(f"Using OpenCL device: {dev.name}")
    print(f"Local size: {args.local_size}")

    prg = cl.Program(ctx, OPENCL_SRC).build()

    rows = []
    for N in args.sizes:
        print(f"\nN={N}")

        row = benchmark_coalesced(
            ctx, queue, prg, N, args.local_size, args.warmup, args.repeat
        )
        rows.append(row)
        print(
            f"  coalesced noop: naive={row['naive_median_ms']:.4f} ms "
            f"opt={row['opt_median_ms']:.4f} ms "
            f"speedup={row['speedup_median']:.3f}x "
            f"correct={row['correct']}"
        )

        row = benchmark_strided(
            ctx, queue, prg, N, args.local_size, args.warmup, args.repeat
        )
        rows.append(row)
        print(
            f"  strided tile:   naive={row['naive_median_ms']:.4f} ms "
            f"opt={row['opt_median_ms']:.4f} ms "
            f"speedup={row['speedup_median']:.3f}x "
            f"correct={row['correct']}"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "kernel",
        "N",
        "local_size",
        "global_size",
        "naive_best_ms",
        "opt_best_ms",
        "naive_median_ms",
        "opt_median_ms",
        "naive_mean_ms",
        "opt_mean_ms",
        "speedup_best",
        "speedup_median",
        "correct",
        "max_abs_err",
    ]

    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()