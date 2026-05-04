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
clang-21 -x cl -cl-std=CL2.0 -Xclang -finclude-default-header \
      --target=nvptx64-nvidia-cuda \
      -O1 -emit-llvm -S "$KERNEL" \
      -o "$OUT/${BASENAME}_nvptx.ll"

echo "==> [2/6] Applying L2 hint transformation (prefetch-hint-pass)"
opt-21 -load-pass-plugin "$BUILD/libPrefetchPass.so" \
    -passes="prefetch-hint-pass" -S \
    "$OUT/${BASENAME}_nvptx.ll" \
    -o "$OUT/${BASENAME}_hint.ll"

echo "==> [3/6] Applying software-pipeline transformation (prefetch-pipeline-pass)"
# loop-simplify inserts a dedicated preheader block when the entry block
# doubles as the loop guard, which our pipeline transform requires.
opt-21 -load-pass-plugin "$BUILD/libPrefetchPass.so" \
    -passes="loop-simplify,prefetch-pipeline-pass" -S \
    "$OUT/${BASENAME}_nvptx.ll" \
    -o "$OUT/${BASENAME}_pipeline.ll"

echo "==> [4/6] Linking OpenCL builtins (get_global_id -> PTX sreg intrinsics)"
llvm-link-21 "$OUT/${BASENAME}_nvptx.ll"   "$BUILTINS" -S -o "$OUT/${BASENAME}_base_linked.ll"
llvm-link-21 "$OUT/${BASENAME}_hint.ll"    "$BUILTINS" -S -o "$OUT/${BASENAME}_hint_linked.ll"
llvm-link-21 "$OUT/${BASENAME}_pipeline.ll" "$BUILTINS" -S -o "$OUT/${BASENAME}_pipeline_linked.ll"

echo "==> [5/6] Lowering baseline to PTX (cpu: $SM)"
llc-21 -march=nvptx64 -mcpu="$SM" "$OUT/${BASENAME}_base_linked.ll" \
    -o "$OUT/${BASENAME}_${SM}_base.ptx"

echo "==> [6/6] Lowering transformed variants to PTX (cpu: $SM)"
llc-21 -march=nvptx64 -mcpu="$SM" "$OUT/${BASENAME}_hint_linked.ll" \
    -o "$OUT/${BASENAME}_${SM}_hint.ptx"
llc-21 -march=nvptx64 -mcpu="$SM" "$OUT/${BASENAME}_pipeline_linked.ll" \
    -o "$OUT/${BASENAME}_${SM}_pipeline.ptx"

echo ""
echo "PTX outputs:"
echo "  Baseline:         $OUT/${BASENAME}_${SM}_base.ptx"
echo "  L2 hint:          $OUT/${BASENAME}_${SM}_hint.ptx"
echo "  Software pipeline: $OUT/${BASENAME}_${SM}_pipeline.ptx"
echo ""
echo "Copy to cluster (only PTX files needed — no LLVM on cluster):"
echo "  scp $OUT/${BASENAME}_${SM}_base.ptx \\"
echo "      $OUT/${BASENAME}_${SM}_hint.ptx \\"
echo "      $OUT/${BASENAME}_${SM}_pipeline.ptx \\"
echo "      benchmarks/bench_prefetch.c  netid@cluster:~/bench/"
echo ""
echo "On the cluster (one-time build, needs only gcc + CUDA):"
echo "  module load cuda"
echo "  gcc bench_prefetch.c -o bench_prefetch -lcuda -I\$CUDA_HOME/include -L\$CUDA_HOME/lib64/stubs -ldl"
echo ""
echo "Profile all three variants with nsys:"
echo "  nsys profile -o report_base     ./bench_prefetch ${BASENAME}_${SM}_base.ptx"
echo "  nsys profile -o report_hint     ./bench_prefetch ${BASENAME}_${SM}_hint.ptx"
echo "  nsys profile -o report_pipeline ./bench_prefetch ${BASENAME}_${SM}_pipeline.ptx"
echo ""
echo "Compare:"
echo "  nsys stats report_base.nsys-rep"
echo "  nsys stats report_hint.nsys-rep"
echo "  nsys stats report_pipeline.nsys-rep"
