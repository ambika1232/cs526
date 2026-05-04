#!/usr/bin/env bash
set -euo pipefail

LLVM_ROOT="${1:-}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT_DIR/build"

mkdir -p "$BUILD_DIR"

if [[ -n "$LLVM_ROOT" ]]; then
  LLVM_DIR="$LLVM_ROOT/lib/cmake/llvm"
elif command -v llvm-config-21 >/dev/null 2>&1; then
  LLVM_DIR="$(llvm-config-21 --cmakedir)"
elif command -v llvm-config >/dev/null 2>&1; then
  LLVM_DIR="$(llvm-config --cmakedir)"
else
  echo "Error: provide LLVM install root or ensure llvm-config is on PATH" >&2
  exit 1
fi

cmake -S "$ROOT_DIR" -B "$BUILD_DIR" -DLLVM_DIR="$LLVM_DIR"
cmake --build "$BUILD_DIR" -j

echo "Built plugin at $BUILD_DIR/libCoalescingPass.so (or platform equivalent)"
