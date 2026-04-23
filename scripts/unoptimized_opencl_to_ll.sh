#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <input.cl> <output.ll> [extra clang args...]" >&2
  exit 1
fi

INPUT="$1"
OUTPUT="$2"
shift 2

clang -x cl -cl-std=CL2.0 -O0 -Xclang -finclude-default-header -emit-llvm -S "$INPUT" -o "$OUTPUT" "$@" 

echo "Wrote LLVM IR to $OUTPUT"
