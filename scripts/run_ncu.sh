#!/usr/bin/env bash
# Usage: bash scripts/run_ncu.sh <bench_binary> <ptx_file> [kernel_name]
# Example: bash scripts/run_ncu.sh ./benchmarks/bench_atax ../ptx_files/atax_sm75_base.ptx
# Default kernel name targets atax_kernel2; pass a third arg to override.

set -euo pipefail

BENCH=${1:?Usage: $0 <bench_binary> <ptx_file> [kernel_name]}
PTX=${2:?Usage: $0 <bench_binary> <ptx_file> [kernel_name]}
KERNEL=${3:-__clang_ocl_kern_imp_atax_kernel2}

ncu \
    --kernel-name "$KERNEL" \
    --metrics smsp__warp_issue_stalled_long_scoreboard_pct,lts__t_sector_hit_rate,l1tex__t_sector_hit_rate,smsp__inst_executed,gpu__dram_throughput \
    "$BENCH" "$PTX"
