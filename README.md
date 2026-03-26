# GPU Memory Coalescing Midterm Repo

This repository is a practical starter kit for an LLVM-based memory coalescing project targeting OpenCL kernels compiled through LLVM IR toward NVPTX. It is designed to support a midterm report even if you do **not** have access to an NVIDIA GPU.

## What this repo provides

- An **LLVM new-pass-manager plugin** that scans memory accesses and classifies simple thread-dependent patterns.
- A lightweight **warp-level coalescing estimator** based on stride and element size.
- Example **OpenCL kernels** with coalesced and non-coalesced patterns.
- Scripts to generate **LLVM IR** from OpenCL kernels.
- A small Python utility to compute and print **estimated memory transactions per warp**.
- A concise **midterm report template** that matches the current state of the project.

## Project idea supported by this repo

The current scope is intentionally realistic for a midterm:

1. Build the OpenCL -> LLVM IR pipeline.
2. Detect memory accesses whose index expression depends on `get_global_id(0)`.
3. Classify simple patterns such as:
   - `A[tid]` -> coalesced
   - `A[tid + c]` -> coalesced
   - `A[tid * k]` -> strided / non-coalesced when `k > 1`
4. Estimate memory transactions at the warp level **without a physical GPU**.

## Repo layout

```text
gpu-coalescing-midterm/
├── CMakeLists.txt
├── README.md
├── docs/
│   └── midterm_report.md
├── include/
│   └── CoalescingAnalysis.h
├── kernels/
│   ├── coalesced.cl
│   ├── offset.cl
│   ├── reverse.cl
│   └── strided.cl
├── sample_ir/
│   └── example_patterns.ll
├── sample_output/
│   └── expected_analysis.txt
├── scripts/
│   ├── build_plugin.sh
│   ├── compile_opencl_to_ll.sh
│   ├── run_analysis.sh
│   └── tx_model.py
├── src/
│   └── CoalescingPass.cpp
└── tests/
    └── notes.md
```

## What the pass currently detects

The pass is intentionally simple and report-friendly. It looks for `getelementptr` instructions and tries to recover the final linear index pattern from the last GEP operand.

Currently supported index forms:

- `%idx = %tid`
- `%idx = add %tid, C`
- `%idx = mul %tid, C`
- `%idx = add (mul %tid, C), C2`

## What it prints

For each GEP-derived memory access, the pass prints:

- base classification
- whether the address is thread dependent
- stride across adjacent threads
- estimated transactions per warp
- a coarse label: `COALESCED`, `LIKELY_COALESCED`, `NON_COALESCED`, or `UNKNOWN`

## Build prerequisites

You need an LLVM build/install that provides:

- `clang`
- `opt`
- `llvm-config`
- LLVM headers and CMake package files

This repo assumes LLVM 15+ and uses the new pass manager plugin interface.

## Build the plugin

```bash
bash scripts/build_plugin.sh /path/to/llvm-install
```

If `llvm-config` is already on your `PATH`, you can also run:

```bash
mkdir -p build && cd build
cmake -DLLVM_DIR=$(llvm-config --cmakedir) ..
cmake --build . -j
```

## Compile OpenCL to LLVM IR

Example:

```bash
bash scripts/compile_opencl_to_ll.sh kernels/strided.cl build/strided.ll
```

Notes:

- This step only generates LLVM IR.
- You do **not** need an NVIDIA GPU for this stage.
- Depending on your local Clang/OpenCL headers, you may need to adjust include paths.

## Run the analysis pass

```bash
bash scripts/run_analysis.sh build/libCoalescingPass.so build/strided.ll
```

## Example result

For `B[tid] = A[tid * 4]` you should expect something like:

```text
[CoalescingPass] function=test
  access=B[tid] thread_dependent=yes stride=1 estimated_transactions=1 class=COALESCED
  access=A[tid*4] thread_dependent=yes stride=4 estimated_transactions=4 class=NON_COALESCED
```

## Static transaction model

The estimator uses a simple model:

- warp size = 32
- segment size = 128 bytes
- element size = 4 bytes for `float`

For a pattern `A[tid * s]`, approximate bytes spanned by a warp as:

```text
span_bytes = ((31 * s) + 1) * element_size
transactions ~= ceil(span_bytes / segment_size)
```

This is not a hardware-perfect model, but it is suitable for a midterm report because it provides a quantitative comparison before and after transformations.

## Suggested midterm positioning

Since you do not have an NVIDIA GPU, present the project as:

- an LLVM-based **static analysis and optimization framework**, and
- an **estimated warp-level memory transaction model** for evaluation.

That is enough for a defensible midterm.

## Natural next steps after midterm

- Use `ScalarEvolution` more systematically
- Rewrite simple strided index forms
- Add benchmarking on a remote GPU or lab machine later
- Explore shared-memory staging only as a stretch goal
