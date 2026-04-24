#!/usr/bin/env bash
# Compiles an OpenCL kernel to NVPTX IR at -O2 (baseline), applies the
# prefetch transformation (treatment), then compiles both to PTX.
#
# Usage: run_gpu_benchmark.sh [kernel.cl] [sm_target]
# Example: run_gpu_benchmark.sh kernels/prefetch.cl sm_80
set -euo pipefail

KERNEL="${1:-kernels/prefetch.cl}"
SM="${2:-sm_89}"  # L40 is Ada Lovelace sm_89
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build"
OUT="$BUILD/bench"

mkdir -p "$OUT"
BASENAME="$(basename "$KERNEL" .cl)"

echo "==> [1/4] Compiling $KERNEL to NVPTX LLVM IR at -O2 (target: nvptx64, cpu: $SM)"
clang -x cl -cl-std=CL2.0 -Xclang -finclude-default-header \
      --target=nvptx64-nvidia-cuda \
      -O2 -emit-llvm -S "$KERNEL" \
      -o "$OUT/${BASENAME}_baseline.ll"

echo "==> [2/4] Applying prefetch transformation"
opt -load-pass-plugin "$BUILD/libPrefetchPass.so" \
    -passes="prefetch-pass" -S \
    "$OUT/${BASENAME}_baseline.ll" \
    -o "$OUT/${BASENAME}_transformed.ll"

echo "==> [3/4] Compiling baseline IR to PTX"
llc -O2 -mcpu="$SM" "$OUT/${BASENAME}_baseline.ll" -o "$OUT/${BASENAME}_baseline.ptx"

echo "==> [4/4] Compiling transformed IR to PTX"
llc -O2 -mcpu="$SM" "$OUT/${BASENAME}_transformed.ll" -o "$OUT/${BASENAME}_transformed.ptx"

echo ""
echo "Outputs:"
echo "  Baseline:    $OUT/${BASENAME}_baseline.ptx"
echo "  Transformed: $OUT/${BASENAME}_transformed.ptx"
echo ""
echo "Copy to cluster:"
echo "  scp $OUT/${BASENAME}_baseline.ptx $OUT/${BASENAME}_transformed.ptx \\"
echo "      benchmarks/bench_prefetch.c  user@cluster:~/bench/"
echo ""
echo "On the cluster (one-time build, needs only gcc + CUDA headers):"
echo "  gcc bench_prefetch.c -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcuda -o bench_prefetch"
echo ""
echo "Profile with nsys:"
echo "  nsys profile -o baseline    ./bench_prefetch ${BASENAME}_baseline.ptx"
echo "  nsys profile -o transformed ./bench_prefetch ${BASENAME}_transformed.ptx"
echo ""
echo "Compare text stats:"
echo "  nsys stats baseline.nsys-rep"
echo "  nsys stats transformed.nsys-rep"
