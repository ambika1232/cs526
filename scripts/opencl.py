#!/usr/bin/env python3
import argparse
import csv
import re
import statistics
from pathlib import Path

import numpy as np
import pyopencl as cl


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def median(xs):
    return statistics.median(xs)


def best(xs):
    return min(xs)


def read_text(path):
    with open(path, "r") as f:
        return f.read()


def find_kernel_names(src):
    """
    Finds OpenCL kernels declared as:
      __kernel void name(...)
      kernel void name(...)
    """
    pat = r"(?:__kernel|kernel)\s+void\s+([A-Za-z_][A-Za-z0-9_]*)\s*\("
    return re.findall(pat, src)


def round_up(x, multiple):
    return ((x + multiple - 1) // multiple) * multiple


def make_context():
    platforms = cl.get_platforms()
    if not platforms:
        raise RuntimeError("No OpenCL platforms found.")

    # Prefer GPU.
    for p in platforms:
        devices = p.get_devices()
        gpus = [d for d in devices if d.type & cl.device_type.GPU]
        if gpus:
            ctx = cl.Context(devices=[gpus[0]])
            queue = cl.CommandQueue(
                ctx,
                properties=cl.command_queue_properties.PROFILING_ENABLE,
            )
            return ctx, queue, gpus[0]

    # Fallback to first available device.
    dev = platforms[0].get_devices()[0]
    ctx = cl.Context(devices=[dev])
    queue = cl.CommandQueue(
        ctx,
        properties=cl.command_queue_properties.PROFILING_ENABLE,
    )
    return ctx, queue, dev


def build_program(ctx, source_path, build_options=""):
    src = read_text(source_path)
    return cl.Program(ctx, src).build(options=build_options)


def time_kernel(queue, kernel, global_size, local_size, warmup, repeat):
    for _ in range(warmup):
        evt = cl.enqueue_nd_range_kernel(queue, kernel, global_size, local_size)
        evt.wait()

    queue.finish()

    times_ms = []
    for _ in range(repeat):
        evt = cl.enqueue_nd_range_kernel(queue, kernel, global_size, local_size)
        evt.wait()

        start = evt.profile.start
        end = evt.profile.end
        times_ms.append((end - start) * 1e-6)

    queue.finish()
    return times_ms


def check_close(a, b, rtol=1e-4, atol=1e-4):
    ok = np.allclose(a, b, rtol=rtol, atol=atol)
    max_abs_err = float(np.max(np.abs(a - b))) if a.size else 0.0
    return bool(ok), max_abs_err


def write_csv(rows, out_path):
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    keys = sorted(set().union(*(r.keys() for r in rows)))

    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------
# Benchmark cases
# ---------------------------------------------------------------------

class BenchmarkCase:
    def __init__(self, name, source_file, kernel_name, run_fn):
        self.name = name
        self.source_file = source_file
        self.kernel_name = kernel_name
        self.run_fn = run_fn


DEFAULT_KERNEL_NAMES = {
    "2mm": "mm2_kernel1",
    "3mm": "mm3_kernel1",
    "atax": "atax_kernel1",
    "bicg": "bicgKernel1",
    "mvt": "mvt_kernel1",
    "nw": "nw_kernel1",
    "rejection_tests": "accept_clean_strided",
}


def kernel_arg_count(kernel):
    try:
        return kernel.num_args
    except AttributeError:
        return kernel.get_info(cl.kernel_info.NUM_ARGS)


def matrix_dim_from_n(N, cap=1024):
    dim = int(np.sqrt(N)) if N > 4096 else N
    return max(16, min(dim, cap))


def run_unary_float_kernel(
    ctx,
    queue,
    program,
    kernel_name,
    N,
    local_size,
    pattern,
):
    """
    Handles simple 1D OpenCL kernels.

    Supports:
      __kernel void test(__global float* A, __global float* B)
    or:
      __kernel void test(__global float* A, __global float* B, int N)

    Important:
      This function returns a keepalive list so PyOpenCL buffers do not get
      garbage-collected before kernel execution.
    """

    mf = cl.mem_flags

    # Allocate conservatively based on expected access patterns.
    if pattern in {"strided", "strided_reuse", "prefetch", "prefetch_thread"}:
        input_len = 8 * N + 65536
    elif pattern in {"offset", "prefetch_loop_invariant"}:
        input_len = 16 * N + 65536
    elif pattern == "reverse":
        input_len = 4 * N + 4096
    else:
        input_len = N + 4096

    A_host = np.random.rand(input_len).astype(np.float32)
    B_host = np.zeros(N, dtype=np.float32)

    A_dev = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A_host)
    B_dev = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=B_host)

    kernel = getattr(program, kernel_name)

    num_args = kernel_arg_count(kernel)

    print(f"    kernel arg count: {num_args}")

    if num_args == 2:
        kernel.set_arg(0, A_dev)
        kernel.set_arg(1, B_dev)

    elif num_args == 3:
        kernel.set_arg(0, A_dev)
        kernel.set_arg(1, B_dev)
        kernel.set_arg(2, np.int32(N))

    else:
        raise RuntimeError(
            f"Unsupported simple kernel arg count: {num_args}. "
            f"Check the signature of {kernel_name}."
        )

    global_size = (round_up(N, local_size),)
    local = (local_size,)

    keepalive = [A_host, B_host, A_dev, B_dev]

    return kernel, global_size, local, B_dev, B_host, keepalive


def run_2mm_kernel(
    ctx,
    queue,
    program,
    kernel_name,
    N,
    local_size,
    pattern,
):
    """
    Generic 2mm-style case.

    This assumes a simplified signature:
      kernel(A, B, C, D, E, N)

    If your 2mm.cl has PolyBench-style arguments such as NI, NJ, NK, NL,
    edit this function to match the exact signature.
    """

    mf = cl.mem_flags

    dim = matrix_dim_from_n(N, cap=1024)

    tmp = np.zeros((dim, dim), dtype=np.float32)
    A = np.random.rand(dim, dim).astype(np.float32)
    B = np.random.rand(dim, dim).astype(np.float32)
    C = np.random.rand(dim, dim).astype(np.float32)
    D = np.random.rand(dim, dim).astype(np.float32)

    tmp_dev = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=tmp)
    A_dev = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    B_dev = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    C_dev = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=C)
    D_dev = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=D)

    kernel = getattr(program, kernel_name)

    num_args = kernel_arg_count(kernel)

    print(f"    kernel arg count: {num_args}")

    if num_args == 9:
        if kernel_name == "mm2_kernel2":
            kernel.set_args(
                tmp_dev,
                C_dev,
                D_dev,
                np.int32(dim),
                np.int32(dim),
                np.int32(dim),
                np.int32(dim),
                np.float32(1.5),
                np.float32(1.2),
            )
            out_dev = D_dev
            out_host = D
        else:
            kernel.set_args(
                tmp_dev,
                A_dev,
                B_dev,
                np.int32(dim),
                np.int32(dim),
                np.int32(dim),
                np.int32(dim),
                np.float32(1.5),
                np.float32(1.2),
            )
            out_dev = tmp_dev
            out_host = tmp
    elif num_args == 6:
        kernel.set_args(
            A_dev,
            B_dev,
            C_dev,
            D_dev,
            tmp_dev,
            np.int32(dim),
        )
        out_dev = tmp_dev
        out_host = tmp
    else:
        raise RuntimeError(
            f"Unsupported 2mm kernel arg count: {num_args}. "
            f"Edit run_2mm_kernel() for the exact signature."
        )

    global_size = (round_up(dim, local_size), round_up(dim, local_size))
    local = (local_size, local_size)

    keepalive = [tmp, A, B, C, D, tmp_dev, A_dev, B_dev, C_dev, D_dev]

    return kernel, global_size, local, out_dev, out_host, keepalive


def run_gemm_kernel(
    ctx,
    queue,
    program,
    kernel_name,
    N,
    local_size,
    pattern,
):
    mf = cl.mem_flags
    dim = matrix_dim_from_n(N, cap=1024)

    A = np.random.rand(dim, dim).astype(np.float32)
    B = np.random.rand(dim, dim).astype(np.float32)
    C = np.random.rand(dim, dim).astype(np.float32)

    A_dev = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    B_dev = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    C_dev = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=C)

    kernel = getattr(program, kernel_name)
    num_args = kernel_arg_count(kernel)
    print(f"    kernel arg count: {num_args}")

    if num_args != 8:
        raise RuntimeError(f"Unsupported gemm arg count: {num_args}")

    kernel.set_args(
        A_dev,
        B_dev,
        C_dev,
        np.float32(1.5),
        np.float32(1.2),
        np.int32(dim),
        np.int32(dim),
        np.int32(dim),
    )

    global_size = (round_up(dim, local_size), round_up(dim, local_size))
    local = (local_size, local_size)
    keepalive = [A, B, C, A_dev, B_dev, C_dev]

    return kernel, global_size, local, C_dev, C, keepalive


def run_3mm_kernel(
    ctx,
    queue,
    program,
    kernel_name,
    N,
    local_size,
    pattern,
):
    mf = cl.mem_flags
    dim = matrix_dim_from_n(N, cap=1024)

    A = np.random.rand(dim, dim).astype(np.float32)
    B = np.random.rand(dim, dim).astype(np.float32)
    C = np.random.rand(dim, dim).astype(np.float32)
    D = np.random.rand(dim, dim).astype(np.float32)
    E = np.zeros((dim, dim), dtype=np.float32)
    F = np.zeros((dim, dim), dtype=np.float32)
    G = np.zeros((dim, dim), dtype=np.float32)

    A_dev = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    B_dev = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    C_dev = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=C)
    D_dev = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=D)
    E_dev = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=E)
    F_dev = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=F)
    G_dev = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=G)

    kernel = getattr(program, kernel_name)
    num_args = kernel_arg_count(kernel)
    print(f"    kernel arg count: {num_args}")

    if num_args != 6:
        raise RuntimeError(f"Unsupported 3mm arg count: {num_args}")

    if kernel_name == "mm3_kernel2":
        kernel.set_args(C_dev, D_dev, F_dev, np.int32(dim), np.int32(dim), np.int32(dim))
        out_dev = F_dev
        out_host = F
    elif kernel_name == "mm3_kernel3":
        kernel.set_args(E_dev, F_dev, G_dev, np.int32(dim), np.int32(dim), np.int32(dim))
        out_dev = G_dev
        out_host = G
    else:
        kernel.set_args(A_dev, B_dev, E_dev, np.int32(dim), np.int32(dim), np.int32(dim))
        out_dev = E_dev
        out_host = E

    global_size = (round_up(dim, local_size), round_up(dim, local_size))
    local = (local_size, local_size)
    keepalive = [A, B, C, D, E, F, G, A_dev, B_dev, C_dev, D_dev, E_dev, F_dev, G_dev]

    return kernel, global_size, local, out_dev, out_host, keepalive


def run_mvt_kernel(
    ctx,
    queue,
    program,
    kernel_name,
    N,
    local_size,
    pattern,
):
    mf = cl.mem_flags
    dim = matrix_dim_from_n(N, cap=4096)

    A = np.random.rand(dim, dim).astype(np.float32)
    x = np.random.rand(dim).astype(np.float32)
    y = np.random.rand(dim).astype(np.float32)

    A_dev = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    x_dev = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=x)
    y_dev = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y)

    kernel = getattr(program, kernel_name)
    num_args = kernel_arg_count(kernel)
    print(f"    kernel arg count: {num_args}")

    if num_args != 4:
        raise RuntimeError(f"Unsupported mvt arg count: {num_args}")

    kernel.set_args(A_dev, x_dev, y_dev, np.int32(dim))

    global_size = (round_up(dim, local_size),)
    local = (local_size,)
    keepalive = [A, x, y, A_dev, x_dev, y_dev]

    return kernel, global_size, local, x_dev, x, keepalive


def run_gesummv_kernel(
    ctx,
    queue,
    program,
    kernel_name,
    N,
    local_size,
    pattern,
):
    mf = cl.mem_flags
    dim = matrix_dim_from_n(N, cap=4096)

    A = np.random.rand(dim, dim).astype(np.float32)
    B = np.random.rand(dim, dim).astype(np.float32)
    x = np.random.rand(dim).astype(np.float32)
    y = np.zeros(dim, dtype=np.float32)
    tmp = np.zeros(dim, dtype=np.float32)

    A_dev = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    B_dev = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    x_dev = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x)
    y_dev = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=y)
    tmp_dev = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=tmp)

    kernel = getattr(program, kernel_name)
    num_args = kernel_arg_count(kernel)
    print(f"    kernel arg count: {num_args}")

    if num_args != 8:
        raise RuntimeError(f"Unsupported gesummv arg count: {num_args}")

    kernel.set_args(
        A_dev,
        B_dev,
        x_dev,
        y_dev,
        tmp_dev,
        np.float32(1.5),
        np.float32(1.2),
        np.int32(dim),
    )

    global_size = (round_up(dim, local_size),)
    local = (local_size,)
    keepalive = [A, B, x, y, tmp, A_dev, B_dev, x_dev, y_dev, tmp_dev]

    return kernel, global_size, local, y_dev, y, keepalive


def run_syrk_kernel(
    ctx,
    queue,
    program,
    kernel_name,
    N,
    local_size,
    pattern,
):
    mf = cl.mem_flags
    dim = matrix_dim_from_n(N, cap=1024)

    A = np.random.rand(dim, dim).astype(np.float32)
    C = np.random.rand(dim, dim).astype(np.float32)

    A_dev = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    C_dev = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=C)

    kernel = getattr(program, kernel_name)
    num_args = kernel_arg_count(kernel)
    print(f"    kernel arg count: {num_args}")

    if num_args != 6:
        raise RuntimeError(f"Unsupported syrk arg count: {num_args}")

    kernel.set_args(
        A_dev,
        C_dev,
        np.float32(1.5),
        np.float32(1.2),
        np.int32(dim),
        np.int32(dim),
    )

    global_size = (round_up(dim, local_size), round_up(dim, local_size))
    local = (local_size, local_size)
    keepalive = [A, C, A_dev, C_dev]

    return kernel, global_size, local, C_dev, C, keepalive


def run_syr2k_kernel(
    ctx,
    queue,
    program,
    kernel_name,
    N,
    local_size,
    pattern,
):
    mf = cl.mem_flags
    dim = matrix_dim_from_n(N, cap=1024)

    A = np.random.rand(dim, dim).astype(np.float32)
    B = np.random.rand(dim, dim).astype(np.float32)
    C = np.random.rand(dim, dim).astype(np.float32)

    A_dev = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    B_dev = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    C_dev = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=C)

    kernel = getattr(program, kernel_name)
    num_args = kernel_arg_count(kernel)
    print(f"    kernel arg count: {num_args}")

    if num_args != 7:
        raise RuntimeError(f"Unsupported syr2k arg count: {num_args}")

    kernel.set_args(
        A_dev,
        B_dev,
        C_dev,
        np.float32(1.5),
        np.float32(1.2),
        np.int32(dim),
        np.int32(dim),
    )

    global_size = (round_up(dim, local_size), round_up(dim, local_size))
    local = (local_size, local_size)
    keepalive = [A, B, C, A_dev, B_dev, C_dev]

    return kernel, global_size, local, C_dev, C, keepalive


def run_convolution_2d_kernel(
    ctx,
    queue,
    program,
    kernel_name,
    N,
    local_size,
    pattern,
):
    """
    Generic 2D convolution case.

    Assumes:
      kernel(input, output, N)

    If your convolution_2d.cl uses width/height/filter args,
    edit this adapter to match the exact signature.
    """

    mf = cl.mem_flags

    dim = matrix_dim_from_n(N, cap=4096)

    inp = np.random.rand(dim, dim).astype(np.float32)
    out = np.zeros((dim, dim), dtype=np.float32)

    inp_dev = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=inp)
    out_dev = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=out)

    kernel = getattr(program, kernel_name)

    num_args = kernel_arg_count(kernel)

    print(f"    kernel arg count: {num_args}")

    if num_args == 3:
        kernel.set_args(inp_dev, out_dev, np.int32(dim))
    elif num_args == 4:
        kernel.set_args(inp_dev, out_dev, np.int32(dim), np.int32(dim))
    else:
        raise RuntimeError(
            f"Unsupported convolution_2d kernel arg count: {num_args}. "
            f"Edit run_convolution_2d_kernel() for the exact signature."
        )

    global_size = (round_up(dim, local_size), round_up(dim, local_size))
    local = (local_size, local_size)

    keepalive = [inp, out, inp_dev, out_dev]

    return kernel, global_size, local, out_dev, out, keepalive


def run_matvec_pair_kernel(
    ctx,
    queue,
    program,
    kernel_name,
    N,
    local_size,
    pattern,
):
    mf = cl.mem_flags
    dim = matrix_dim_from_n(N, cap=4096)

    A = np.random.rand(dim, dim).astype(np.float32)
    x = np.random.rand(dim).astype(np.float32)
    y = np.zeros(dim, dtype=np.float32)

    A_dev = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    x_dev = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x)
    y_dev = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=y)

    kernel = getattr(program, kernel_name)
    num_args = kernel_arg_count(kernel)
    print(f"    kernel arg count: {num_args}")

    if num_args != 5:
        raise RuntimeError(f"Unsupported {pattern} arg count: {num_args}")

    kernel.set_args(A_dev, x_dev, y_dev, np.int32(dim), np.int32(dim))

    global_size = (round_up(dim, local_size),)
    local = (local_size,)
    keepalive = [A, x, y, A_dev, x_dev, y_dev]

    return kernel, global_size, local, y_dev, y, keepalive


# ---------------------------------------------------------------------
# Main benchmark logic
# ---------------------------------------------------------------------

def benchmark_pair(
    ctx,
    queue,
    case,
    original_path,
    optimized_path,
    N,
    local_size,
    warmup,
    repeat,
    build_options,
):
    prog_orig = build_program(ctx, original_path, build_options)
    prog_opt = build_program(ctx, optimized_path, build_options)

    # Use identical random inputs for original and optimized kernels.
    np.random.seed(12345)
    k_orig, g_orig, l_orig, out_orig_dev, out_orig_host, keep_orig = case.run_fn(
        ctx,
        queue,
        prog_orig,
        case.kernel_name,
        N,
        local_size,
        case.name,
    )

    np.random.seed(12345)
    k_opt, g_opt, l_opt, out_opt_dev, out_opt_host, keep_opt = case.run_fn(
        ctx,
        queue,
        prog_opt,
        case.kernel_name,
        N,
        local_size,
        case.name,
    )

    t_orig = time_kernel(queue, k_orig, g_orig, l_orig, warmup, repeat)
    t_opt = time_kernel(queue, k_opt, g_opt, l_opt, warmup, repeat)

    # Keep buffers alive through execution and copy-back.
    _ = keep_orig, keep_opt

    cl.enqueue_copy(queue, out_orig_host, out_orig_dev).wait()
    cl.enqueue_copy(queue, out_opt_host, out_opt_dev).wait()

    correct, max_abs_err = check_close(out_orig_host, out_opt_host)

    row = {
        "benchmark": case.name,
        "kernel_name": case.kernel_name,
        "N": N,
        "local_size": local_size,
        "original_file": str(original_path),
        "optimized_file": str(optimized_path),
        "original_best_ms": best(t_orig),
        "optimized_best_ms": best(t_opt),
        "original_median_ms": median(t_orig),
        "optimized_median_ms": median(t_opt),
        "speedup_best": best(t_orig) / best(t_opt),
        "speedup_median": median(t_orig) / median(t_opt),
        "correct": correct,
        "max_abs_err": max_abs_err,
    }

    return row


def infer_kernel_name(source_path, requested_name=None, kernel_basename=None):
    src = read_text(source_path)
    names = find_kernel_names(src)

    if requested_name:
        if requested_name not in names:
            print(
                f"Warning: requested kernel '{requested_name}' "
                f"not found in {source_path}."
            )
            print(f"Found kernels: {names}")
        return requested_name

    if len(names) == 1:
        return names[0]

    default_name = DEFAULT_KERNEL_NAMES.get(kernel_basename)
    if default_name and default_name in names:
        print(
            f"Multiple kernels found in {source_path}; "
            f"defaulting to '{default_name}'."
        )
        return default_name

    if not names:
        raise RuntimeError(f"No OpenCL kernels found in {source_path}")

    raise RuntimeError(
        f"Multiple kernels found in {source_path}: {names}. "
        f"Pass --kernel-name explicitly."
    )


def make_case(kernel, source_path, kernel_name):
    simple_1d = {
        "coalesced",
        "strided",
        "strided_reuse",
        "offset",
        "reverse",
        "prefetch",
        "prefetch_thread",
        "prefetch_loop_invariant",
        "pipeline_bench",
        "bounded_kernels",
        "rejection_tests",
    }

    if kernel in simple_1d:
        return BenchmarkCase(
            kernel,
            source_path,
            kernel_name,
            run_unary_float_kernel,
        )
    if kernel == "pipeline_bench":
        return BenchmarkCase(
            kernel,
            source_path,
            kernel_name,
            run_pipeline_bench_kernel,
        )

    if kernel == "2mm":
        return BenchmarkCase(kernel, source_path, kernel_name, run_2mm_kernel)

    if kernel == "3mm":
        return BenchmarkCase(kernel, source_path, kernel_name, run_3mm_kernel)

    if kernel == "gemm":
        return BenchmarkCase(kernel, source_path, kernel_name, run_gemm_kernel)

    if kernel == "convolution_2d":
        return BenchmarkCase(kernel, source_path, kernel_name, run_convolution_2d_kernel)

    if kernel in {"atax", "bicg"}:
        return BenchmarkCase(kernel, source_path, kernel_name, run_matvec_pair_kernel)

    if kernel == "mvt":
        return BenchmarkCase(kernel, source_path, kernel_name, run_mvt_kernel)

    if kernel == "gesummv":
        return BenchmarkCase(kernel, source_path, kernel_name, run_gesummv_kernel)

    if kernel == "syrk":
        return BenchmarkCase(kernel, source_path, kernel_name, run_syrk_kernel)

    if kernel == "syr2k":
        return BenchmarkCase(kernel, source_path, kernel_name, run_syr2k_kernel)

    raise RuntimeError(
        f"No adapter for kernel '{kernel}'. "
        f"Add a run_{kernel}_kernel function for its argument signature."
    )

def run_pipeline_bench_kernel(
    ctx,
    queue,
    program,
    kernel_name,
    N,
    local_size,
    pattern,
):
    """
    Adapter for pipeline_bench.

    Assumes one of:
      pipeline_bench(__global float* A, __global float* B, int N)
    or:
      pipeline_bench(__global float* A, __global float* B, __global float* C)

    Adjust after checking the actual .cl signature.
    """

    mf = cl.mem_flags

    input_len = 16 * N + 65536

    A_host = np.random.rand(input_len).astype(np.float32)
    B_host = np.zeros(input_len, dtype=np.float32)
    C_host = np.zeros(input_len, dtype=np.float32)

    A_dev = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=A_host)
    B_dev = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=B_host)
    C_dev = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=C_host)

    kernel = getattr(program, kernel_name)

    try:
        num_args = kernel.num_args
    except AttributeError:
        num_args = kernel.get_info(cl.kernel_info.NUM_ARGS)

    print(f"    kernel arg count: {num_args}")

    if num_args != 3:
        raise RuntimeError(
            f"pipeline_bench expected 3 args in this adapter, got {num_args}"
        )

    # Try the common A, B, N signature first.
    # If grep shows the third arg is a buffer, change this to C_dev.
    kernel.set_arg(0, A_dev)
    kernel.set_arg(1, B_dev)
    kernel.set_arg(2, np.int32(N))

    global_size = (round_up(N, local_size),)
    local = (local_size,)

    keepalive = [A_host, B_host, C_host, A_dev, B_dev, C_dev]

    return kernel, global_size, local, B_dev, B_host[:N], keepalive

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--kernel",
        required=True,
        help=(
            "Kernel basename, e.g. strided, coalesced, reverse, "
            "offset, 2mm, convolution_2d."
        ),
    )

    ap.add_argument(
        "--kernel-name",
        default=None,
        help=(
            "OpenCL function name. If omitted, inferred when file has "
            "exactly one kernel."
        ),
    )

    ap.add_argument(
        "--kernels-dir",
        default="kernels",
    )

    ap.add_argument(
        "--opt-dir",
        default="kernels_opt",
        help="Directory containing optimized/transformed .cl files.",
    )

    ap.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[1 << 18, 1 << 20, 1 << 22, 1 << 24],
    )

    ap.add_argument(
        "--local-size",
        type=int,
        default=256,
    )

    ap.add_argument(
        "--warmup",
        type=int,
        default=10,
    )

    ap.add_argument(
        "--repeat",
        type=int,
        default=50,
    )

    ap.add_argument(
        "--build-options",
        default="",
    )

    ap.add_argument(
        "--out",
        default="bench_results/opencl_kernel_speedups.csv",
    )

    args = ap.parse_args()

    original_path = Path(args.kernels_dir) / f"{args.kernel}.cl"
    optimized_path = Path(args.opt_dir) / f"{args.kernel}.cl"

    if not original_path.exists():
        raise FileNotFoundError(f"Missing original kernel: {original_path}")

    if not optimized_path.exists():
        raise FileNotFoundError(
            f"Missing optimized kernel: {optimized_path}\n"
            f"Create this file first, e.g. {optimized_path}"
        )

    kernel_name = infer_kernel_name(
        original_path,
        args.kernel_name,
        args.kernel,
    )
    case = make_case(args.kernel, original_path, kernel_name)

    ctx, queue, dev = make_context()

    print("OpenCL device:", dev.name)
    print("Original:", original_path)
    print("Optimized:", optimized_path)
    print("Kernel function:", kernel_name)

    rows = []

    for N in args.sizes:
        print(f"\nN={N}")

        row = benchmark_pair(
            ctx=ctx,
            queue=queue,
            case=case,
            original_path=original_path,
            optimized_path=optimized_path,
            N=N,
            local_size=args.local_size,
            warmup=args.warmup,
            repeat=args.repeat,
            build_options=args.build_options,
        )

        rows.append(row)

        print(
            f"  original={row['original_median_ms']:.4f} ms "
            f"optimized={row['optimized_median_ms']:.4f} ms "
            f"speedup={row['speedup_median']:.3f}x "
            f"correct={row['correct']} "
            f"max_abs_err={row['max_abs_err']:.3e}"
        )

    write_csv(rows, args.out)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
