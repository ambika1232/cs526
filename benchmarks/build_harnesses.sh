#!/usr/bin/env bash
# Compiles all bench_*.c harnesses against the CUDA Driver API.
# Run this once before run_benchmarks.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${CUDA_HOME:-}" ]]; then
    CUDA_ROOT="$CUDA_HOME"
elif [[ -n "${CUDA_PATH:-}" ]]; then
    CUDA_ROOT="$CUDA_PATH"
elif command -v nvcc >/dev/null 2>&1; then
    CUDA_ROOT="$(dirname "$(dirname "$(which nvcc)")")"
elif [[ -n "${CONDA_PREFIX:-}" ]]; then
    CUDA_ROOT="$CONDA_PREFIX"
else
    echo "ERROR: Cannot find CUDA. Set CUDA_HOME or activate the llvm-dev conda env." >&2
    exit 1
fi

if [[ ! -f "$CUDA_ROOT/include/cuda.h" && -f "$CUDA_ROOT/targets/x86_64-linux/include/cuda.h" ]]; then
    CUDA_ROOT="$CUDA_ROOT/targets/x86_64-linux"
fi

CUDA_LIB="${CUDA_ROOT}/lib64/stubs"
[[ -d "$CUDA_LIB" ]] || CUDA_LIB="${CUDA_ROOT}/lib/stubs"
[[ -d "$CUDA_LIB" ]] || CUDA_LIB="${CUDA_ROOT}/lib"

echo "CUDA root: $CUDA_ROOT"
echo "CUDA lib:  $CUDA_LIB"
echo ""

for SRC in "$SCRIPT_DIR"/bench_*.c; do
    BIN="${SRC%.c}"
    gcc "$SRC" -o "$BIN" -lcuda -I"$CUDA_ROOT/include" -L"$CUDA_LIB" -ldl
    echo "  compiled $(basename "$BIN")"
done

echo ""
echo "All harnesses built. Run benchmarks with:"
echo "  bash run_benchmarks.sh"
