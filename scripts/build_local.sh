#!/usr/bin/env bash
# Builds the LLVM plugin and generates PTX for all major kernels.
# Runs entirely on the local machine (Jupyter VM) — no SCP step.
#
# Usage: bash scripts/build_local.sh [sm_target]
#   sm_target — e.g. sm_80 (default) or sm_89

set -euo pipefail

SM="${1:-sm_80}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

KERNELS=(gemm atax bicg 2mm 3mm mvt syrk syr2k gesummv pipeline_bench)

echo "==> Building LLVM plugin"
bash "$ROOT/scripts/build_plugin.sh"

echo ""
echo "==> Generating PTX for all kernels (target: $SM)"
for KERNEL in "${KERNELS[@]}"; do
    CL="$ROOT/kernels/${KERNEL}.cl"
    if [[ ! -f "$CL" ]]; then
        echo "SKIP $KERNEL — $CL not found"
        continue
    fi
    echo "--- $KERNEL"
    bash "$ROOT/scripts/run_gpu_benchmark.sh" "$CL" "$SM"
done

echo ""
echo "PTX files written to $ROOT/build/bench/"
echo ""
echo "Run benchmarks with:"
echo "  cd $ROOT/benchmarks"
echo "  bash run_nsys.sh $SM ../build/bench"
