#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KERNEL_DIR="$ROOT/kernels"
BUILD_DIR="$ROOT/build"
OUT_DIR="$ROOT/bench_results"

PLUGIN="$BUILD_DIR/CoalescingPass.so"
if [[ ! -f "$PLUGIN" ]]; then
  PLUGIN="$BUILD_DIR/libCoalescingPass.so"
fi

mkdir -p "$OUT_DIR"/{bc,ll,logs}

if [[ ! -f "$PLUGIN" ]]; then
  echo "ERROR: plugin not found: $PLUGIN"
  echo "Run: cd $ROOT/build && cmake --build . -j"
  exit 1
fi

if ! command -v clang >/dev/null; then
  echo "ERROR: clang not found. Activate llvm-dev first."
  exit 1
fi

if ! command -v opt >/dev/null; then
  echo "ERROR: opt not found. Activate llvm-dev first."
  exit 1
fi

if ! command -v llvm-as >/dev/null; then
  echo "ERROR: llvm-as not found. Activate llvm-dev first."
  exit 1
fi

CSV="$OUT_DIR/summary.csv"

echo "kernel,compile_ms,analyze_before_ms,transform_ms,analyze_after_ms,baseline_instrs,transformed_instrs,before_unknown,before_coalesced,before_strided,before_invariant,before_tile_actions,before_keep_actions,after_unknown,after_coalesced,after_strided,after_invariant,after_tile_actions,after_keep_actions,transform_success" > "$CSV"

count_instrs() {
  local ll="$1"

  if [[ ! -f "$ll" ]]; then
    echo 0
    return
  fi

  local n
  n=$(grep -E '^[[:space:]]*(%[A-Za-z0-9_.-]+[[:space:]]*=|store |br |ret |call |invoke )' "$ll" 2>/dev/null | wc -l | tr -d ' ' || true)
  echo "${n:-0}"
}

count_pattern() {
  local file="$1"
  local pattern="$2"

  if [[ ! -f "$file" ]]; then
    echo 0
    return
  fi

  local n
  n=$(grep -E "$pattern" "$file" 2>/dev/null | wc -l | tr -d ' ' || true)
  echo "${n:-0}"
}

now_ns() {
  date +%s%N
}

elapsed_ms() {
  local start="$1"
  local end="$2"
  echo $(( (end - start) / 1000000 ))
}

compile_kernel() {
  local src="$1"
  local bc="$2"
  local ll="$3"

  clang -cc1 -x cl -cl-std=CL2.0 \
    -triple spir64-unknown-unknown \
    -emit-llvm-bc -O0 \
    -finclude-default-header \
    -fdeclare-opencl-builtins \
    -disable-O0-optnone \
    "$src" -o "$bc"

  llvm-dis "$bc" -o "$ll"
}

analyze_kernel() {
  local input="$1"
  local log="$2"

  # The SPIR target-machine warning is harmless for your current analysis.
  opt -load-pass-plugin "$PLUGIN" \
    -passes='function(coalescing-pass)' \
    -disable-output \
    "$input" > "$log" 2>&1 || true
}

transform_kernel() {
  local input="$1"
  local outll="$2"
  local log="$3"

  opt -load-pass-plugin "$PLUGIN" \
    -passes='function(tile-remap-pass)' \
    -S "$input" -o "$outll" > "$log" 2>&1
}

for src in "$KERNEL_DIR"/*.cl; do
  kernel="$(basename "$src" .cl)"

  echo "=== Benchmarking $kernel ==="

  base_bc="$OUT_DIR/bc/${kernel}.bc"
  base_ll="$OUT_DIR/ll/${kernel}.baseline.ll"
  transformed_ll="$OUT_DIR/ll/${kernel}.transformed.ll"
  transformed_bc="$OUT_DIR/bc/${kernel}.transformed.bc"

  before_log="$OUT_DIR/logs/${kernel}.before.log"
  transform_log="$OUT_DIR/logs/${kernel}.transform.log"
  after_log="$OUT_DIR/logs/${kernel}.after.log"

  start=$(now_ns)
  if ! compile_kernel "$src" "$base_bc" "$base_ll" > "$OUT_DIR/logs/${kernel}.compile.log" 2>&1; then
    echo "  compile failed; see $OUT_DIR/logs/${kernel}.compile.log"
    continue
  fi
  end=$(now_ns)
  compile_ms=$(elapsed_ms "$start" "$end")

  start=$(now_ns)
  analyze_kernel "$base_bc" "$before_log"
  end=$(now_ns)
  analyze_before_ms=$(elapsed_ms "$start" "$end")

  transform_success=1
  start=$(now_ns)
  if ! transform_kernel "$base_bc" "$transformed_ll" "$transform_log"; then
    transform_success=0
    cp "$base_ll" "$transformed_ll"
  fi
  end=$(now_ns)
  transform_ms=$(elapsed_ms "$start" "$end")

  if [[ "$transform_success" -eq 1 ]]; then
    llvm-as "$transformed_ll" -o "$transformed_bc" 2>> "$transform_log" || {
      transform_success=0
      cp "$base_bc" "$transformed_bc"
    }
  else
    cp "$base_bc" "$transformed_bc"
  fi

  start=$(now_ns)
  analyze_kernel "$transformed_bc" "$after_log"
  end=$(now_ns)
  analyze_after_ms=$(elapsed_ms "$start" "$end")

  baseline_instrs=$(count_instrs "$base_ll")
  transformed_instrs=$(count_instrs "$transformed_ll")

  before_unknown=$(count_pattern "$before_log" 'class=UNKNOWN')
  before_coalesced=$(count_pattern "$before_log" 'class=(FULLY_COALESCED|COALESCED_BUT_MISALIGNED|PARTIALLY_COALESCED|LIKELY_COALESCED)')
  before_strided=$(count_pattern "$before_log" 'class=STRIDED')
  before_invariant=$(count_pattern "$before_log" 'class=(THREAD_INVARIANT|INVARIANT_ACROSS_X)')
  before_tile_actions=$(count_pattern "$before_log" 'action=TILE_REMAP')
  before_keep_actions=$(count_pattern "$before_log" 'action=KEEP_GLOBAL')

  after_unknown=$(count_pattern "$after_log" 'class=UNKNOWN')
  after_coalesced=$(count_pattern "$after_log" 'class=(FULLY_COALESCED|COALESCED_BUT_MISALIGNED|PARTIALLY_COALESCED|LIKELY_COALESCED)')
  after_strided=$(count_pattern "$after_log" 'class=STRIDED')
  after_invariant=$(count_pattern "$after_log" 'class=(THREAD_INVARIANT|INVARIANT_ACROSS_X)')
  after_tile_actions=$(count_pattern "$after_log" 'action=TILE_REMAP')
  after_keep_actions=$(count_pattern "$after_log" 'action=KEEP_GLOBAL')

  echo "$kernel,$compile_ms,$analyze_before_ms,$transform_ms,$analyze_after_ms,$baseline_instrs,$transformed_instrs,$before_unknown,$before_coalesced,$before_strided,$before_invariant,$before_tile_actions,$before_keep_actions,$after_unknown,$after_coalesced,$after_strided,$after_invariant,$after_tile_actions,$after_keep_actions,$transform_success" >> "$CSV"

  echo "  done: before_log=$before_log"
  echo "        transform_log=$transform_log"
  echo "        after_log=$after_log"
done

echo
echo "Wrote: $CSV"
echo "View with:"
echo "  column -s, -t $CSV | less -S"
