# GPU Memory Coalescing 

This repository is a practical starter kit for an LLVM-based memory coalescing project targeting OpenCL kernels compiled through LLVM IR toward NVPTX.

- An **LLVM new-pass-manager plugin** that scans memory accesses and classifies simple thread-dependent patterns.
- A lightweight **warp-level coalescing estimator** based on stride and element size.
- Example **OpenCL kernels** with  coalesced and non-coalesced patterns.
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
cs526/
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

## Create and load `llvm-dev`

Run all of these setup, build, and benchmark commands from the NCSA JupyterLab
environment at:

```text
https://jupyter.ncsa.illinois.edu
```

The paths below use environment variables so the instructions work for any
student account and do not depend on a specific course year or semester.

The JupyterLab container used for this project does not have the cluster module
system, so `module load` is not available:

```bash
module: command not found
```

Use a user-local conda environment instead. Create it once with LLVM and Clang
from conda-forge:

```bash
source /opt/conda/etc/profile.d/conda.sh
conda create -n llvm-dev -c conda-forge llvmdev clang cmake make compilers -y
conda activate llvm-dev
```

This creates the environment under:

```text
$HOME/.conda/envs/llvm-dev
```

The important tools come from that environment:

```text
$CONDA_PREFIX/bin/clang
$CONDA_PREFIX/bin/opt
$CONDA_PREFIX/bin/llvm-config
```

In the working setup, both LLVM and Clang reported version `22.1.4`.

For future sessions, load the environment with:

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate llvm-dev
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

which clang opt llvm-config
clang --version
opt --version
llvm-config --version
```

The key point is that conda installs a complete user-local LLVM toolchain, so
the project does not need `module load`, root access, or a system-wide LLVM
install.

For the Python benchmark and plotting scripts, install the Python-side runtime
packages into the same environment:

```bash
conda install -n llvm-dev -c conda-forge numpy pyopencl matplotlib -y
```

### CUDA runtime for GPU benchmarks

The LLVM pass build only needs LLVM and Clang. Runtime GPU benchmarking also
needs CUDA runtime libraries when using CUDA-backed tools such as CuPy. If
`nvidia-smi` works but `nvcc`, `libnvrtc.so`, or `libcurand.so` are missing,
install the runtime pieces into the same environment:

```bash
conda install -c nvidia cuda-nvrtc cuda-cudart -y
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

This provides libraries such as:

```text
$CONDA_PREFIX/lib/libnvrtc.so.12
$CONDA_PREFIX/lib/libcudart.so.12
```

## Build the plugin

From the repository root:

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate llvm-dev
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

rm -rf build
mkdir build
cd build
cmake .. -DLLVM_DIR="$(llvm-config --cmakedir)"
cmake --build . -j
```

`llvm-config --cmakedir` should point inside the conda environment, for example:

```text
$CONDA_PREFIX/lib/cmake/llvm
```

That directory contains LLVM's CMake package files, which is why CMake can find
LLVM.

You can also use the helper script after activating `llvm-dev`:

```bash
bash scripts/build_plugin.sh
```

Depending on the CMake/LLVM setup, the plugin is written as either
`build/CoalescingPass.so` or `build/libCoalescingPass.so`.

### LLVM 22 source compatibility

The conda environment uses LLVM 22, so the pass source has to use LLVM 22 API
names and header paths. In this repo that means using:

- `#include "llvm/Plugins/PassPlugin.h"` instead of
  `#include "llvm/Passes/PassPlugin.h"`
- `PointerType::get(...)` instead of removed typed-pointer helpers such as
  `Type::getInt8PtrTy(...)`
- `Intrinsic::getOrInsertDeclaration(...)` where LLVM 22 requires it instead
  of older `Intrinsic::getDeclaration(...)` usage

## Compile OpenCL to LLVM IR

Example:

```bash
bash scripts/compile_opencl_to_ll.sh kernels/strided.cl build/strided.ll
```

## Run the analysis pass

```bash
PLUGIN=build/CoalescingPass.so
test -f "$PLUGIN" || PLUGIN=build/libCoalescingPass.so
bash scripts/run_analysis.sh "$PLUGIN" build/strided.ll
```

The registered pass names are:

- `coalescing-pass`: prints memory-access classifications.
- `coalescing-rewrite-pass`: applies conservative local reuse rewrites.
- `tile-remap-pass`: rewrites selected strided loads through local memory when
  the pass decides the pattern is profitable.

To run the full static pass benchmark across kernels in `kernels/`:

```bash
bash scripts/benchmark_kernels.sh
```

This writes:

- `bench_results/summary.csv`: compile/analyze/transform timings and access
  classification counts.
- `bench_results/logs/*.before.log`: analysis before transformation.
- `bench_results/logs/*.transform.log`: `tile-remap-pass` output.
- `bench_results/logs/*.after.log`: analysis after transformation.
- `bench_results/ll/*.baseline.ll` and `bench_results/ll/*.transformed.ll`:
  generated LLVM IR.

## Run the runtime benchmark and create plots

The main OpenCL benchmark harness compares a baseline kernel in `kernels/`
against the matching optimized kernel in `kernels_opt/`.

Run one benchmark:

```bash
python scripts/opencl.py \
  --kernel gemm \
  --sizes 65536 262144 1048576 \
  --local-size 16 \
  --warmup 5 \
  --repeat 20 \
  --out bench_results/opencl_all/gemm.csv
```

Run the supported benchmark set and create the plots in one command:

```bash
python scripts/plot_opencl_results.py \
  --run \
  --input-dir bench_results/opencl_all \
  --glob '*.csv' \
  --out-dir bench_results/plots \
  --sizes 65536 262144 1048576 \
  --warmup 5 \
  --repeat 20
```

Regenerate plots from existing CSVs without rerunning the benchmarks:

```bash
python scripts/plot_opencl_results.py \
  --input-dir bench_results/opencl_all \
  --glob '*.csv' \
  --out-dir bench_results/plots
```

The main outputs are:

- `bench_results/plots/plot_summary.csv`
- `bench_results/plots/best_speedup_by_kernel.png`
- `bench_results/plots/geomean_speedup_by_kernel.png`
- `bench_results/plots/speedup_by_kernel.png`
- `bench_results/plots/speedup_heatmap.png`
- `bench_results/plots/runtime_median_N*.png`

## Example result

For `B[tid] = A[tid * 4]` you should expect:

```text
[CoalescingPass] function=test
  access=B[tid] thread_dependent=yes stride=1 estimated_transactions=1 class=COALESCED
  access=A[tid*4] thread_dependent=yes stride=4 estimated_transactions=4 class=NON_COALESCED
```
