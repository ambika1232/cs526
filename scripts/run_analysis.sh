#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <plugin.so> <input.ll>" >&2
  exit 1
fi

PLUGIN="$1"
INPUT="$2"

opt -load-pass-plugin "$PLUGIN" -passes="coalescing-pass" -disable-output "$INPUT" 2>&1
