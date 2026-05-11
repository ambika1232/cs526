#!/usr/bin/env bash
# Runs all bench harnesses directly (no nsys) and writes timing to a CSV.
# Each bench binary uses CUDA events internally, so no profiler is needed.
#
# Usage: bash run_benchmarks.sh [sm_target] [ptx_dir] [N]
#   sm_target — e.g. sm_80 (default)
#   ptx_dir   — directory containing PTX files (default: ../build/bench)
#   N         — matrix dimension for most kernels (default: 4096)
#
# pipeline_bench uses a larger iteration count automatically (see PB_N below).

set -euo pipefail

SM="${1:-sm_80}"
PTX_DIR="${2:-../build/bench}"
N="${3:-4096}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS="$SCRIPT_DIR/results"
CSV="$RESULTS/timings_${SM}_N${N}.csv"

mkdir -p "$RESULTS"

# pipeline_bench working set must exceed A100 L2 (40 MB); 65536 * 256 * 4 = 64 MB
case "$SM" in
    sm_89) PB_N=131072 ;;
    sm_80) PB_N=65536  ;;
    *)     PB_N=32768  ;;
esac

KERNELS=(gemm atax bicg 2mm 3mm mvt syrk syr2k gesummv pipeline_bench)
VARIANTS=(base hint pipeline)

echo "SM=$SM  PTX_DIR=$PTX_DIR  N=$N  PB_N=$PB_N"
echo ""
echo "kernel,variant,N,avg_ms" > "$CSV"

for KERNEL in "${KERNELS[@]}"; do
    BENCH="$SCRIPT_DIR/bench_${KERNEL}"
    if [[ ! -x "$BENCH" ]]; then
        echo "SKIP $KERNEL — bench_${KERNEL} not built"
        continue
    fi

    KERNEL_N="$N"
    [[ "$KERNEL" == "pipeline_bench" ]] && KERNEL_N="$PB_N"

    for VARIANT in "${VARIANTS[@]}"; do
        PTX="$PTX_DIR/${KERNEL}_${SM}_${VARIANT}.ptx"
        if [[ ! -f "$PTX" ]]; then
            echo "SKIP ${KERNEL}/${VARIANT} — PTX not found: $PTX"
            continue
        fi

        printf "  %-20s %-10s N=%-8s" "${KERNEL}" "${VARIANT}" "${KERNEL_N}"
        OUTPUT="$("$BENCH" "$PTX" "$KERNEL_N" 2>&1)"
        AVG_MS="$(echo "$OUTPUT" | grep -oP 'avg kernel time: \K[0-9.]+')"

        if [[ -z "$AVG_MS" ]]; then
            echo "ERROR — no timing output. Full output:"
            echo "$OUTPUT"
            continue
        fi

        printf "%s ms\n" "$AVG_MS"
        echo "${KERNEL},${VARIANT},${KERNEL_N},${AVG_MS}" >> "$CSV"
    done
done

echo ""
echo "Results written to $CSV"
echo ""
echo "Generate plots with:"
echo "  python3 plot_timings.py $CSV"
