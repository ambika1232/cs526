#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <plugin.so> <pass-name> <input.ll>" >&2
  exit 1
fi

PLUGIN="$1"
PASS_NAME="$2"  
INPUT="$3"

opt -load-pass-plugin "$PLUGIN" -passes="mem2reg,$PASS_NAME" -disable-output "$INPUT" 2>&1