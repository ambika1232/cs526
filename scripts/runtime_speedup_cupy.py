#!/usr/bin/env python3
import argparse
import csv
import statistics
from pathlib import Path

import cupy as cp
import numpy as np


CUDA_SRC = r'''
extern "C" __global__
void coalesced_naive(const float* A, float* B, int N) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < N) {
        B[gid] = A[gid];
    }
}

extern "C" __global__
void strided_naive(const float* A, float* B, int N) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < N) {
        B[gid] = A[4 * gid];
    }
}

extern "C" __global__
void strided_tile_remap(const float* A, float* B, int N) {
    extern __shared__ float tile[];

    int lid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int block_base_out = blockIdx.x * blockDim.x;

    int dense_base = 4 * block_base_out;
    int logical_tile_elems = 4 * (blockDim.x - 1) + 1;

    for (int off = lid; off < logical_tile_elems; off += blockDim.x) {
        int gidx = dense_base + off;
        if (gidx < 4 * N) {
            tile[off] = A[gidx];
        } else {
            tile[off] = 0.0f;
        }
    }

    __syncthreads();

    if (gid < N) {
        B[gid] = tile[4 * lid];
    }
}
'''


def median(xs):
    return statistics.median(xs)


def best(xs):
    return min(xs)


def time_kernel(kernel, grid, block, args, shared_mem=0, warmup=10, repeat=50):
    for _ in range(warmup):
        kernel(grid, block, args, shared_mem=shared_mem)
    cp.cuda.Stream.null.synchronize()

    times = []
    start = cp.cuda.Event()
    stop = cp.cuda.Event()

    for _ in range(repeat):
        start.record()
        kernel(grid, block, args, shared_mem=shared_mem)
        stop.record()
        stop.synchronize()
        times.append(cp.cuda.get_elapsed_time(start, stop))

    cp.cuda.Stream.null.synchronize()
    return times


def bench_coalesced(N, block, warmup, repeat, k_coal):
    # A = cp.random.random(N, dtype=cp.float32)
    A = cp.asarray(np.random.random(N).astype(np.float32))
    B1 = cp.zeros(N, dtype=cp.float32)
    B2 = cp.zeros(N, dtype=cp.float32)

    grid = ((N + block - 1) // block,)

    t1 = time_kernel(
        k_coal,
        grid,
        (block,),
        (A, B1, np.int32(N)),
        warmup=warmup,
        repeat=repeat,
    )
    t2 = time_kernel(
        k_coal,
        grid,
        (block,),
        (A, B2, np.int32(N)),
        warmup=warmup,
        repeat=repeat,
    )

    correct = bool(cp.allclose(B1, B2, rtol=1e-5, atol=1e-6).get())

    return {
        "kernel": "coalesced_noop",
        "N": N,
        "block": block,
        "naive_best_ms": best(t1),
        "opt_best_ms": best(t2),
        "naive_median_ms": median(t1),
        "opt_median_ms": median(t2),
        "speedup_best": best(t1) / best(t2),
        "speedup_median": median(t1) / median(t2),
        "correct": correct,
    }


def bench_strided(N, block, warmup, repeat, k_naive, k_tile):
    # A = cp.random.random(4 * N, dtype=cp.float32)
    A = cp.asarray(np.random.random(4 * N).astype(np.float32))
    B1 = cp.zeros(N, dtype=cp.float32)
    B2 = cp.zeros(N, dtype=cp.float32)

    grid = ((N + block - 1) // block,)
    logical_tile_elems = 4 * (block - 1) + 1
    shared_mem = logical_tile_elems * cp.dtype(cp.float32).itemsize

    t1 = time_kernel(
        k_naive,
        grid,
        (block,),
        (A, B1, np.int32(N)),
        warmup=warmup,
        repeat=repeat,
    )
    t2 = time_kernel(
        k_tile,
        grid,
        (block,),
        (A, B2, np.int32(N)),
        shared_mem=shared_mem,
        warmup=warmup,
        repeat=repeat,
    )

    correct = bool(cp.allclose(B1, B2, rtol=1e-5, atol=1e-6).get())
    max_abs_err = float(cp.max(cp.abs(B1 - B2)).get())

    return {
        "kernel": "strided_tile_remap",
        "N": N,
        "block": block,
        "naive_best_ms": best(t1),
        "opt_best_ms": best(t2),
        "naive_median_ms": median(t1),
        "opt_median_ms": median(t2),
        "speedup_best": best(t1) / best(t2),
        "speedup_median": median(t1) / median(t2),
        "correct": correct,
        "max_abs_err": max_abs_err,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", type=int,
                    default=[1 << 18, 1 << 20, 1 << 22, 1 << 24, 1 << 26])
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeat", type=int, default=50)
    ap.add_argument("--out", default="bench_results/runtime_speedups_cupy.csv")
    args = ap.parse_args()

    print("Device:", cp.cuda.runtime.getDeviceProperties(0)["name"])

    module = cp.RawModule(code=CUDA_SRC, options=("--std=c++11",))
    k_coal = module.get_function("coalesced_naive")
    k_strided = module.get_function("strided_naive")
    k_tile = module.get_function("strided_tile_remap")

    rows = []
    for N in args.sizes:
        print(f"\nN={N}")

        r = bench_coalesced(N, args.block, args.warmup, args.repeat, k_coal)
        rows.append(r)
        print(
            f"  coalesced noop: naive={r['naive_median_ms']:.4f} ms "
            f"opt={r['opt_median_ms']:.4f} ms "
            f"speedup={r['speedup_median']:.3f}x correct={r['correct']}"
        )

        r = bench_strided(N, args.block, args.warmup, args.repeat, k_strided, k_tile)
        rows.append(r)
        print(
            f"  strided tile:   naive={r['naive_median_ms']:.4f} ms "
            f"opt={r['opt_median_ms']:.4f} ms "
            f"speedup={r['speedup_median']:.3f}x correct={r['correct']}"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    keys = sorted(set().union(*(r.keys() for r in rows)))
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
