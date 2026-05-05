#!/usr/bin/env python3
"""
Analyze register pressure and occupancy impact across PTX variants.

Usage:
    python3 scripts/analyze_registers.py [--arch sm_75] [--block-size 256]
    python3 scripts/analyze_registers.py --arch sm_89 --block-size 256

Reports: register count, spill bytes, and occupancy (threads/SM) for each
kernel function across base / hint / pipeline PTX variants.

Occupancy thresholds are computed for the given block size and arch.
"""

import subprocess, re, sys, argparse, os
from pathlib import Path
from collections import defaultdict

# ── SM register file sizes ──────────────────────────────────────────────────
SM_REGS = {
    "sm_75": 65536,   # Turing (Quadro RTX 6000)
    "sm_80": 65536,   # Ampere (A100)
    "sm_89": 65536,   # Ada Lovelace (L40S)
}
SM_MAX_THREADS = {
    "sm_75": 1024,
    "sm_80": 1024,
    "sm_89": 1024,
}
SM_MAX_BLOCKS = {
    "sm_75": 32,
    "sm_80": 32,
    "sm_89": 32,
}
# Register allocation granularity (regs per thread, rounded up to this)
SM_REG_GRANULARITY = {
    "sm_75": 8,
    "sm_80": 8,
    "sm_89": 8,
}

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch", default="sm_75",
                   help="PTX architecture target (default: sm_75)")
    p.add_argument("--block-size", type=int, default=256,
                   help="Threads per block used at launch (default: 256)")
    p.add_argument("--ptx-dir", default=None,
                   help="Directory with PTX files (default: build/bench/ relative to repo root)")
    return p.parse_args()


def repo_root():
    here = Path(__file__).resolve().parent
    return here.parent


def compute_occupancy(regs, arch, block_size):
    """Return max threads/SM given register count per thread."""
    if arch not in SM_REGS:
        return None, None
    total_regs = SM_REGS[arch]
    max_threads = SM_MAX_THREADS[arch]
    max_blocks = SM_MAX_BLOCKS[arch]
    gran = SM_REG_GRANULARITY.get(arch, 8)

    # round up register count to allocation granularity
    regs_alloc = ((regs + gran - 1) // gran) * gran
    regs_per_block = regs_alloc * block_size

    if regs_per_block == 0:
        return max_threads, max_threads

    blocks_by_regs = total_regs // regs_per_block
    blocks_by_threads = max_threads // block_size
    active_blocks = min(blocks_by_regs, blocks_by_threads, max_blocks)
    active_threads = active_blocks * block_size
    pct = 100.0 * active_threads / max_threads
    return active_threads, pct


def ptxas_info(ptx_path, arch):
    """Run ptxas -v and return list of (func_name, regs, spill_stores, spill_loads)."""
    try:
        result = subprocess.run(
            ["ptxas", f"-arch={arch}", "-v", str(ptx_path), "-o", "/dev/null"],
            capture_output=True, text=True
        )
        output = result.stderr  # ptxas prints info to stderr
        if result.returncode != 0 and not output.strip():
            output = result.stderr
        if "is not defined for option" in output or "fatal" in output.lower():
            return None  # caller will warn once
    except FileNotFoundError:
        print("ERROR: ptxas not found. Install CUDA toolkit.", file=sys.stderr)
        sys.exit(1)

    entries = []
    func = None
    spill_stores = spill_loads = 0
    for line in output.splitlines():
        # "Compiling entry function 'foo' for 'sm_75'"
        m = re.search(r"Compiling entry function '(.+?)' for '", line)
        if m:
            func = m.group(1)
            spill_stores = spill_loads = 0
            continue
        # "    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads"
        m = re.search(r"(\d+) bytes spill stores.*?(\d+) bytes spill loads", line)
        if m and func:
            spill_stores = int(m.group(1))
            spill_loads = int(m.group(2))
            continue
        # "    Used 12 registers, 384 bytes cmem[0]"
        m = re.search(r"Used (\d+) registers", line)
        if m and func:
            regs = int(m.group(1))
            entries.append((func, regs, spill_stores, spill_loads))
            func = None

    return entries


def short_func(name):
    """Strip compiler-generated prefix for display."""
    # __clang_ocl_kern_imp_atax_kernel2 → atax_kernel2
    name = re.sub(r"^__clang_ocl_kern_imp_", "", name)
    # _Z13get_global_idj → get_global_id (skip helper)
    name = re.sub(r"^_Z\d+", "", name)
    return name


def is_entry_kernel(name):
    """True for the actual kernel entry points (not helpers like get_global_id)."""
    return "__clang_ocl_kern_imp_" in name


def main():
    args = parse_args()
    arch = args.arch
    block_size = args.block_size

    ptx_dir = Path(args.ptx_dir) if args.ptx_dir else repo_root() / "build" / "bench"
    if not ptx_dir.exists():
        print(f"ERROR: PTX directory not found: {ptx_dir}", file=sys.stderr)
        sys.exit(1)

    # Find all PTX files for this arch: <kernel>_<arch>_<variant>.ptx
    # arch string in filename uses underscores: sm_75 → sm_75
    arch_tag = arch.replace("-", "_")  # sm_75 stays sm_75
    pattern = f"*_{arch_tag}_*.ptx"
    ptx_files = sorted(ptx_dir.glob(pattern))
    if not ptx_files:
        print(f"No PTX files matching '{pattern}' in {ptx_dir}", file=sys.stderr)
        sys.exit(1)

    # Group by kernel name and variant
    # filename: atax_sm_75_base.ptx → kernel=atax, variant=base
    file_map = defaultdict(dict)
    for f in ptx_files:
        stem = f.stem  # e.g. atax_sm_75_base
        # strip arch tag from stem
        rest = stem.replace(f"_{arch_tag}_", "_")  # atax_base
        parts = rest.rsplit("_", 1)
        if len(parts) != 2:
            continue
        kernel, variant = parts
        if variant in ("base", "hint", "pipeline"):
            file_map[kernel][variant] = f

    VARIANTS = ["base", "hint", "pipeline"]
    COLORS = {
        "green": "\033[92m",
        "red":   "\033[91m",
        "yellow":"\033[93m",
        "reset": "\033[0m",
    }
    use_color = sys.stdout.isatty()

    def color(text, c):
        if not use_color:
            return text
        return f"{COLORS[c]}{text}{COLORS['reset']}"

    max_threads = SM_MAX_THREADS.get(arch, 1024)

    print(f"\n{'='*80}")
    print(f"  Register & Occupancy Analysis — {arch}, block_size={block_size}")
    print(f"  PTX dir: {ptx_dir}")
    print(f"{'='*80}\n")

    header = f"{'Kernel':<30} {'Variant':<10} {'Regs':>5} {'SpillSt':>8} {'SpillLd':>8} {'Threads/SM':>12} {'Occupancy':>10}"
    print(header)
    print("-" * len(header))

    occupancy_drops = []
    arch_warned = False

    for kernel in sorted(file_map.keys()):
        variants = file_map[kernel]
        prev_occ = {}  # func → (threads, pct) for base

        first = True
        for variant in VARIANTS:
            if variant not in variants:
                continue
            result = ptxas_info(variants[variant], arch)
            if result is None:
                if not arch_warned:
                    print(f"WARNING: ptxas cannot compile {arch} with local CUDA toolkit "
                          f"(needs CUDA 11.8+). Copy PTX to cluster and run there.",
                          file=sys.stderr)
                    arch_warned = True
                continue
            # Only show actual kernel entry points
            entries = [(f, r, ss, sl) for f, r, ss, sl in result if is_entry_kernel(f)]

            for func, regs, spill_stores, spill_loads in entries:
                threads, pct = compute_occupancy(regs, arch, block_size)
                fname = short_func(func)

                occ_str = f"{threads} ({pct:.0f}%)" if threads else "?"

                spill_flag = ""
                if spill_stores > 0 or spill_loads > 0:
                    spill_flag = " !"

                row = f"{'  ' + fname if not first else fname:<30} {variant:<10} {regs:>5} {spill_stores:>8} {spill_loads:>8} {occ_str:>12}"
                row += spill_flag

                # Color: compare pipeline vs base occupancy
                if variant == "base":
                    prev_occ[func] = (threads, pct)
                elif variant == "pipeline" and func in prev_occ:
                    base_threads, base_pct = prev_occ[func]
                    if threads is not None and base_threads is not None and threads < base_threads:
                        drop = base_threads - threads
                        row += color(f"  ← OCCUPANCY DROP -{drop} threads/SM", "red")
                        occupancy_drops.append((kernel, fname, base_threads, threads, base_pct, pct))
                    elif threads == base_threads:
                        pass  # no change

                print(row)
                first = False
        print()

    # Summary
    if occupancy_drops:
        print(color("OCCUPANCY DROPS DETECTED (pipeline pass):", "red"))
        for kernel, func, base_t, pipe_t, base_p, pipe_p in occupancy_drops:
            print(f"  {kernel}/{func}: {base_t} → {pipe_t} threads/SM  ({base_p:.0f}% → {pipe_p:.0f}%)")
        print()
    else:
        print(color("No occupancy drops from pipeline pass.", "green"))

    print(f"Note: threshold for full occupancy ({max_threads} threads/SM) with {block_size}-thread blocks on {arch}:")
    _, pct_full = compute_occupancy(1, arch, block_size)
    # Find boundary register count
    prev_threads = max_threads
    for r in range(1, 256):
        t, _ = compute_occupancy(r, arch, block_size)
        if t < prev_threads:
            print(f"  >{r-1} regs/thread → drops to {t} threads/SM")
            prev_threads = t
        if t <= max_threads // 4:
            break
    print()


if __name__ == "__main__":
    main()
