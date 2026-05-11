#!/usr/bin/env bash
# Runs nsys profiling for all major kernels (base/hint/pipeline variants).
# Outputs .nsys-rep files to benchmarks/reports/.
#
# Usage: bash run_nsys.sh [sm_target] [ptx_dir] [N]
#   sm_target  — e.g. sm_75 (default) or sm_89
#   ptx_dir    — directory containing the PTX files (default: ../build/bench)
#   N          — matrix dimension NxN (default: 4096); use larger values (e.g. 8192)
#                to exceed L2 cache on GPUs with large L2 (e.g. L40S has 96MB L2;
#                atax/bicg working set = N^2*4 bytes, needs N>5017 to spill from L2)
#
# Example (from benchmarks/):
#   bash run_nsys.sh sm_89 ./ptx_files 8192

set -euo pipefail

SM="${1:-sm_80}"
PTX_DIR="${2:-../build/bench}"
N="${3:-4096}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPORTS="$SCRIPT_DIR/reports"

mkdir -p "$REPORTS"

# Recompile all bench harnesses to pick up any source changes (e.g. new N argument)
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

for KERNEL in "${KERNELS[@]}"; do
    BENCH="$SCRIPT_DIR/bench_${KERNEL}"
    if [[ ! -x "$BENCH" ]]; then
        echo "SKIP $KERNEL — bench_${KERNEL} not built"
        continue
    fi

    for VARIANT in "${VARIANTS[@]}"; do
        PTX="$PTX_DIR/${KERNEL}_${SM}_${VARIANT}.ptx"
        if [[ ! -f "$PTX" ]]; then
            echo "SKIP ${KERNEL} ${VARIANT} — PTX not found: $PTX"
            continue
        fi

        REPORT="$REPORTS/report_${KERNEL}_${VARIANT}_N${N}"
        echo "==> nsys profile: ${KERNEL} / ${VARIANT} (N=${N})"
        nsys profile --force-overwrite true -o "$REPORT" "$BENCH" "$PTX" "$N"
        echo "    -> $REPORT.nsys-rep"
    done
done

# ── pipeline_bench: synthetic kernel, uses loop-iteration count not matrix dim ──
# Working set = 256 * n * 4 bytes; must exceed L2 to force DRAM traffic.
#   sm_75  (6 MB L2):  n=32768  → 32 MB
#   sm_80  (40 MB L2): n=65536  → 64 MB
#   sm_89  (96 MB L2): n=131072 → 128 MB
case "$SM" in
    sm_80) PB_N=65536  ;;
    sm_89) PB_N=131072 ;;
    *)     PB_N=32768  ;;   # sm_75 and anything else
esac

BENCH="$SCRIPT_DIR/bench_pipeline_bench"
if [[ -x "$BENCH" ]]; then
    for VARIANT in "${VARIANTS[@]}"; do
        PTX="$PTX_DIR/pipeline_bench_${SM}_${VARIANT}.ptx"
        if [[ ! -f "$PTX" ]]; then
            echo "SKIP pipeline_bench ${VARIANT} — PTX not found: $PTX"
            continue
        fi
        REPORT="$REPORTS/report_pipeline_bench_${VARIANT}_N${PB_N}"
        echo "==> nsys profile: pipeline_bench / ${VARIANT} (n=${PB_N})"
        nsys profile --force-overwrite true -o "$REPORT" "$BENCH" "$PTX" "$PB_N"
        echo "    -> $REPORT.nsys-rep"
    done
else
    echo "SKIP pipeline_bench — bench_pipeline_bench not built"
fi

echo ""
echo "All reports written to $REPORTS/"
echo ""
echo "To view results:"
for KERNEL in "${KERNELS[@]}"; do
    for VARIANT in "${VARIANTS[@]}"; do
        echo "  nsys stats $REPORTS/report_${KERNEL}_${VARIANT}_N${N}.nsys-rep 2>&1 | grep -A 15 'CUDA GPU Kernel Summary'"
    done
done
