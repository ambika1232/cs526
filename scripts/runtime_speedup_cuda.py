#!/usr/bin/env python3
import argparse
import csv
import statistics
from pathlib import Path

import numpy as np
from numba import cuda


@cuda.jit
def strided_naive(A, B, N):
    gid = cuda.grid(1)
    if gid < N:
        B[gid] = A[4 * gid]


@cuda.jit
def strided_tile_remap(A, B, N):
    # Fixed max local tile for blockDim.x <= 256.
    tile = cuda.shared.array(shape=1021, dtype=numba_float32)

    tid = cuda.threadIdx.x
    gid = cuda.grid(1)
    block_base_out = cuda.blockIdx.x * cuda.blockDim.x
    dense_base = 4 * block_base_out
    logical_tile_elems = 4 * (cuda.blockDim.x - 1) + 1

    off = tid
    while off < logical_tile_elems:
        gidx = dense_base + off
        if gidx < 4 * N:
            tile[off] = A[gidx]
        else:
            tile[off] = 0.0
        off += cuda.blockDim.x

    cuda.syncthreads()

    if gid < N:
        B[gid] = tile[4 * tid]


@cuda.jit
def coalesced_naive(A, B, N):
    gid = cuda.grid(1)
    if gid < N:
        B[gid] = A[gid]


def median_ms(times):
    return statistics.median(times)


def best_ms(times):
    return min(times)


def time_kernel(kernel, grid, block, args, warmup, repeat):
    for _ in range(warmup):
        kernel[grid, block](*args)
    cuda.synchronize()

    times = []
    start = cuda.event()
    end = cuda.event()

    for _ in range(repeat):
        start.record()
        kernel[grid, block](*args)
        end.record()
        end.synchronize()
        times.append(cuda.event_elapsed_time(start, end))

    cuda.synchronize()
    return times


def bench_strided(N, block, warmup, repeat):
    A = np.random.rand(4 * N).astype(np.float32)
    B1 = np.zeros(N, dtype=np.float32)
    B2 = np.zeros(N, dtype=np.float32)

    dA = cuda.to_device(A)
    dB1 = cuda.to_device(B1)
    dB2 = cuda.to_device(B2)

    grid = (N + block - 1) // block

    t_naive = time_kernel(strided_naive, grid, block, (dA, dB1, N), warmup, repeat)
    t_opt = time_kernel(strided_tile_remap, grid, block, (dA, dB2, N), warmup, repeat)

    out1 = dB1.copy_to_host()
    out2 = dB2.copy_to_host()

    correct = np.allclose(out1, out2, rtol=1e-5, atol=1e-6)
    max_abs_err = float(np.max(np.abs(out1 - out2)))

    return {
        "kernel": "strided_tile_remap",
        "N": N,
        "block": block,
        "naive_best_ms": best_ms(t_naive),
        "opt_best_ms": best_ms(t_opt),
        "naive_median_ms": median_ms(t_naive),
        "opt_median_ms": median_ms(t_opt),
        "speedup_best": best_ms(t_naive) / best_ms(t_opt),
        "speedup_median": median_ms(t_naive) / median_ms(t_opt),
        "correct": correct,
        "max_abs_err": max_abs_err,
    }


def bench_coalesced(N, block, warmup, repeat):
    A = np.random.rand(N).astype(np.float32)
    B1 = np.zeros(N, dtype=np.float32)
    B2 = np.zeros(N, dtype=np.float32)

    dA = cuda.to_device(A)
    dB1 = cuda.to_device(B1)
    dB2 = cuda.to_device(B2)

    grid = (N + block - 1) // block

    t_naive = time_kernel(coalesced_naive, grid, block, (dA, dB1, N), warmup, repeat)
    t_opt = time_kernel(coalesced_naive, grid, block, (dA, dB2, N), warmup, repeat)

    out1 = dB1.copy_to_host()
    out2 = dB2.copy_to_host()

    correct = np.allclose(out1, out2, rtol=1e-5, atol=1e-6)
    max_abs_err = float(np.max(np.abs(out1 - out2)))

    return {
        "kernel": "coalesced_noop",
        "N": N,
        "block": block,
        "naive_best_ms": best_ms(t_naive),
        "opt_best_ms": best_ms(t_opt),
        "naive_median_ms": median_ms(t_naive),
        "opt_median_ms": median_ms(t_opt),
        "speedup_best": best_ms(t_naive) / best_ms(t_opt),
        "speedup_median": median_ms(t_naive) / median_ms(t_opt),
        "correct": correct,
        "max_abs_err": max_abs_err,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int,
                        default=[1 << 18, 1 << 20, 1 << 22, 1 << 24])
    parser.add_argument("--block", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--out", default="bench_results/runtime_speedups_cuda.csv")
    args = parser.parse_args()

    if not cuda.is_available():
        raise RuntimeError("CUDA is not available. Run this on a GPU node.")

    print("Using CUDA device:", cuda.get_current_device().name)

    rows = []
    for N in args.sizes:
        print(f"\nN={N}")
        row = bench_coalesced(N, args.block, args.warmup, args.repeat)
        rows.append(row)
        print(f"  coalesced noop: {row['speedup_median']:.3f}x correct={row['correct']}")

        row = bench_strided(N, args.block, args.warmup, args.repeat)
        rows.append(row)
        print(f"  strided tile:   {row['speedup_median']:.3f}x correct={row['correct']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", newline="") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {out}")


if __name__ == "__main__":
    from numba import float32 as numba_float32
    main()
