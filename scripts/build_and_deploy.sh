#!/usr/bin/env bash
# Builds the LLVM plugin, generates PTX for all major kernels, and scps to cluster.
#
# Usage: bash scripts/build_and_deploy.sh [sm_target]
#   sm_target — e.g. sm_75 (default) or sm_80
#
# Fill in CLUSTER_DEST before running.

set -euo pipefail

SM="${1:-sm_75}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --- Fill this in ---
CLUSTER_DEST="tamirb2@cc-login.campuscluster.illinois.edu:~/scratch/tamirb2/cs526/benchmarks/ptx_files/"
# --------------------

KERNELS=(gemm atax bicg 2mm 3mm mvt syrk syr2k gesummv)

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
echo "==> Collecting PTX files"
PTX_FILES=()
for KERNEL in "${KERNELS[@]}"; do
    for VARIANT in base hint pipeline; do
        PTX="$ROOT/build/bench/${KERNEL}_${SM}_${VARIANT}.ptx"
        if [[ -f "$PTX" ]]; then
            PTX_FILES+=("$PTX")
        else
            echo "WARNING: missing $PTX"
        fi
    done
done

echo ""
echo "==> Copying ${#PTX_FILES[@]} PTX files to $CLUSTER_DEST"
scp "${PTX_FILES[@]}" "$CLUSTER_DEST"

echo ""
echo "Done. On the cluster, rebuild bench binaries if needed:"
echo "  module load cuda"
echo "  for f in bench_*.c; do gcc \$f -o \${f%.c} -lcuda -I\$CUDA_HOME/include -L\$CUDA_HOME/lib64/stubs -ldl; done"
echo "  (or just let run_nsys.sh recompile them automatically)"
echo ""
echo "Then profile with:"
echo "  bash run_nsys.sh $SM ."
