#!/usr/bin/env bash
# Compiles an OpenCL kernel to NVPTX IR, applies the prefetch transformation,
# links OpenCL builtins, and lowers both variants to PTX for A100 (sm_80).
#
# Usage: run_gpu_benchmark.sh [kernel.cl] [sm_target]
# Example: run_gpu_benchmark.sh kernels/prefetch.cl sm_80
set -euo pipefail

KERNEL="${1:-kernels/prefetch.cl}"
SM="${2:-sm_80}"  # A100 is sm_80
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build"
OUT="$BUILD/bench"
BUILTINS="$ROOT/scripts/opencl_nvptx_builtins.ll"

mkdir -p "$OUT"
BASENAME="$(basename "$KERNEL" .cl)"

echo "==> [1/5] Compiling $KERNEL to NVPTX LLVM IR at -O1 (target: nvptx64, cpu: $SM)"
# -O1: keeps loop structure intact for ScalarEvolution (SCEV) in PrefetchPass.
# -O2 would inline/unroll loops before our pass sees them.
clang -x cl -cl-std=CL2.0 -Xclang -finclude-default-header \
      --target=nvptx64-nvidia-cuda \
      -O1 -emit-llvm -S "$KERNEL" \
      -o "$OUT/${BASENAME}_nvptx.ll"

echo "==> [2/5] Applying prefetch transformation"
opt -load-pass-plugin "$BUILD/libPrefetchPass.so" \
    -passes="prefetch-pass" -S \
    "$OUT/${BASENAME}_nvptx.ll" \
    -o "$OUT/${BASENAME}_transformed.ll"

echo "==> [3/5] Linking OpenCL builtins (get_global_id -> PTX sreg intrinsics)"
llvm-link "$OUT/${BASENAME}_nvptx.ll"     "$BUILTINS" -S -o "$OUT/${BASENAME}_base_linked.ll"
llvm-link "$OUT/${BASENAME}_transformed.ll" "$BUILTINS" -S -o "$OUT/${BASENAME}_prefetch_linked.ll"

echo "==> [4/5] Lowering baseline to PTX (cpu: $SM)"
llc -march=nvptx64 -mcpu="$SM" "$OUT/${BASENAME}_base_linked.ll" \
    -o "$OUT/${BASENAME}_${SM}_base.ptx"

echo "==> [5/5] Lowering transformed to PTX (cpu: $SM)"
llc -march=nvptx64 -mcpu="$SM" "$OUT/${BASENAME}_prefetch_linked.ll" \
    -o "$OUT/${BASENAME}_${SM}_prefetch.ptx"

echo ""
echo "PTX outputs:"
echo "  Baseline:    $OUT/${BASENAME}_${SM}_base.ptx"
echo "  Transformed: $OUT/${BASENAME}_${SM}_prefetch.ptx"
echo ""
echo "Copy to cluster (only the PTX files needed — no LLVM on cluster):"
echo "  scp $OUT/${BASENAME}_${SM}_base.ptx $OUT/${BASENAME}_${SM}_prefetch.ptx \\"
echo "      benchmarks/bench_prefetch.c  netid@cluster:~/bench/"
echo ""
echo "On the cluster (one-time build, needs only gcc + CUDA):"
echo "  module load cuda"
echo "  gcc bench_prefetch.c -o bench_prefetch -lcuda -I\$CUDA_HOME/include -L\$CUDA_HOME/lib64/stubs -ldl"
echo ""
echo "Profile with nsys:"
echo "  nsys profile -o report_base     ./bench_prefetch ${BASENAME}_${SM}_base.ptx"
echo "  nsys profile -o report_prefetch ./bench_prefetch ${BASENAME}_${SM}_prefetch.ptx"
echo ""
echo "Compare:"
echo "  nsys stats report_base.nsys-rep"
echo "  nsys stats report_prefetch.nsys-rep"
