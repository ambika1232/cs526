#!/usr/bin/env bash
# Runs nsys profiling for all kernels across a sweep of N values.
# Produces one .nsys-rep per (kernel, variant, N) triple.
#
# Usage: bash run_nsys_sweep.sh [sm_target] [ptx_dir] [N_list...]
#   sm_target  — e.g. sm_75 (default) or sm_80 / sm_89
#   ptx_dir    — directory containing PTX files (default: ../build/bench)
#   N_list     — space-separated matrix dimensions to sweep
#                (default: 4 8 16 32 64 128 256 512 1024 2048 4096 8192)
#
# Examples (from benchmarks/):
#   bash run_nsys_sweep.sh
#   bash run_nsys_sweep.sh sm_89 ./ptx_files
#   bash run_nsys_sweep.sh sm_75 ./ptx_files 256 512 1024 2048 4096 8192

set -euo pipefail

SM="${1:-sm_80}"
PTX_DIR="${2:-../build/bench}"
shift 2 2>/dev/null || shift $# 2>/dev/null || true

if [[ $# -gt 0 ]]; then
    N_VALUES=("$@")
else
    N_VALUES=(4 8 16 32 64 128 256 512 1024 2048 4096 8192)
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPORTS="$SCRIPT_DIR/reports"
mkdir -p "$REPORTS"

# ── compile bench harnesses once ─────────────────────────────────────────────
if [[ -n "${CUDA_HOME:-}" ]]; then
    CUDA_ROOT="$CUDA_HOME"
elif [[ -n "${CUDA_PATH:-}" ]]; then
    CUDA_ROOT="$CUDA_PATH"
elif command -v nvcc >/dev/null 2>&1; then
    CUDA_ROOT="$(dirname "$(dirname "$(which nvcc)")")"
elif [[ -n "${CONDA_PREFIX:-}" ]]; then
    if [[ -d "$CONDA_PREFIX/targets/x86_64-linux" ]]; then
        CUDA_ROOT="$CONDA_PREFIX/targets/x86_64-linux"
    else
        CUDA_ROOT="$CONDA_PREFIX"
    fi
else
    echo "ERROR: Cannot find CUDA. Set CUDA_HOME or activate the llvm-dev conda env." >&2
    exit 1
fi
CUDA_LIB="${CUDA_ROOT}/lib64/stubs"
[[ -d "$CUDA_LIB" ]] || CUDA_LIB="${CUDA_ROOT}/lib"
echo "==> Recompiling bench harnesses (CUDA: $CUDA_ROOT)..."
for SRC in "$SCRIPT_DIR"/bench_*.c; do
    BIN="${SRC%.c}"
    gcc "$SRC" -o "$BIN" -lcuda -I"$CUDA_ROOT/include" -L"$CUDA_LIB" -ldl
    echo "    compiled $(basename "$BIN")"
done

KERNELS=(gemm atax bicg 2mm 3mm mvt syrk syr2k gesummv)
VARIANTS=(base hint pipeline)

TOTAL=$(( ${#KERNELS[@]} * ${#VARIANTS[@]} * ${#N_VALUES[@]} ))
DONE=0

echo ""
echo "==> Sweeping N in: ${N_VALUES[*]}"
echo "    Kernels: ${KERNELS[*]}"
echo "    Variants: ${VARIANTS[*]}"
echo "    Total profiling runs: $TOTAL"
echo ""

for N in "${N_VALUES[@]}"; do
    echo "──────────────────────────────────────────"
    echo "  N = $N"
    echo "──────────────────────────────────────────"

    for KERNEL in "${KERNELS[@]}"; do
        BENCH="$SCRIPT_DIR/bench_${KERNEL}"
        if [[ ! -x "$BENCH" ]]; then
            echo "  SKIP $KERNEL — bench_${KERNEL} not built"
            continue
        fi

        for VARIANT in "${VARIANTS[@]}"; do
            PTX="$PTX_DIR/${KERNEL}_${SM}_${VARIANT}.ptx"
            if [[ ! -f "$PTX" ]]; then
                echo "  SKIP ${KERNEL}/${VARIANT} — PTX not found: $PTX"
                continue
            fi

            REPORT="$REPORTS/report_${KERNEL}_${VARIANT}_N${N}"
            DONE=$(( DONE + 1 ))
            echo "  [$DONE/$TOTAL] nsys: ${KERNEL}/${VARIANT} N=${N}"
            nsys profile --force-overwrite true -o "$REPORT" "$BENCH" "$PTX" "$N" \
                2>/dev/null
        done
    done
done

echo ""
echo "==> All reports written to $REPORTS/"
echo ""
echo "To plot results across all N values:"
echo "  python3 $SCRIPT_DIR/plot_sweep.py $REPORTS --out $SCRIPT_DIR/sweep_results.png"
