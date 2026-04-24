#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <input.ll> <output.ll>" >&2
  exit 1
fi

INPUT="$1"
OUTPUT="$2"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

opt -load-pass-plugin "$ROOT/build/libPrefetchPass.so" \
    -passes="prefetch-pass" -S \
    "$INPUT" -o "$OUTPUT"

echo "Wrote transformed IR to $OUTPUT"
