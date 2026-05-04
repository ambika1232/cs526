#!/usr/bin/env bash
# Runs nsys profiling for all major kernels (base/hint/pipeline variants).
# Outputs .nsys-rep files to benchmarks/reports/.
#
# Usage: bash run_nsys.sh [sm_target] [ptx_dir]
#   sm_target  — e.g. sm_75 (default) or sm_80
#   ptx_dir    — directory containing the PTX files (default: ../build/bench)
#
# Example (from benchmarks/):
#   bash run_nsys.sh sm_75 ../build/bench

set -euo pipefail

SM="${1:-sm_75}"
PTX_DIR="${2:-../build/bench}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPORTS="$SCRIPT_DIR/reports"

mkdir -p "$REPORTS"

KERNELS=(gemm atax bicg 2mm 3mm)
VARIANTS=(base hint pipeline)

for KERNEL in "${KERNELS[@]}"; do
    BENCH="$SCRIPT_DIR/bench_${KERNEL}"
    if [[ ! -x "$BENCH" ]]; then
        echo "SKIP $KERNEL — bench_${KERNEL} not built (run: make bench_${KERNEL})"
        continue
    fi

    for VARIANT in "${VARIANTS[@]}"; do
        PTX="$PTX_DIR/${KERNEL}_${SM}_${VARIANT}.ptx"
        if [[ ! -f "$PTX" ]]; then
            echo "SKIP ${KERNEL} ${VARIANT} — PTX not found: $PTX"
            continue
        fi

        REPORT="$REPORTS/report_${KERNEL}_${VARIANT}"
        echo "==> nsys profile: ${KERNEL} / ${VARIANT}"
        nsys profile -o "$REPORT" "$BENCH" "$PTX"
        echo "    -> $REPORT.nsys-rep"
    done
done

echo ""
echo "All reports written to $REPORTS/"
echo ""
echo "To view results:"
for KERNEL in "${KERNELS[@]}"; do
    for VARIANT in "${VARIANTS[@]}"; do
        echo "  nsys stats $REPORTS/report_${KERNEL}_${VARIANT}.nsys-rep 2>&1 | grep -A 15 'CUDA GPU Kernel Summary'"
    done
done
