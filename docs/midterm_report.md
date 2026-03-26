# Midterm Progress Report: Memory Coalescing in LLVM (NVPTX)

## Project Overview
This project studies compiler-level memory coalescing optimization for OpenCL workloads compiled through LLVM's NVPTX backend. The goal of the first phase is to detect thread-dependent memory access patterns in LLVM IR and estimate whether they are likely to be coalesced at the warp level. Since I do not currently have access to an NVIDIA GPU, the current evaluation is based on static analysis of LLVM IR and a simple warp-level memory transaction model.

## Progress
During the first stage, I completed the LLVM/NVPTX environment setup and established a working OpenCL -> LLVM IR pipeline. I then built a prototype LLVM pass that scans `getelementptr`-based memory accesses and classifies simple thread-indexed patterns. The pass currently handles patterns such as `A[tid]`, `A[tid + c]`, and `A[tid * k]`, where `tid` is derived from `get_global_id(0)`.

To support quantitative evaluation without hardware, I also implemented a lightweight transaction estimator that approximates how many 128-byte memory segments a warp would touch for a given stride pattern. Under this model, contiguous accesses such as `A[tid]` are estimated to require one transaction per warp, while strided accesses such as `A[tid * 4]` require multiple transactions.

## Preliminary Results
The current implementation correctly distinguishes between simple coalesced and non-coalesced synthetic kernels. For example, the pass classifies `A[tid]` as coalesced and `A[tid * 4]` as non-coalesced. This establishes a working baseline for future transformations and provides early evidence that the analysis can identify relevant memory patterns in LLVM IR.

## Next Steps
The next phase is to extend the analysis to more complex affine expressions using ScalarEvolution and begin implementing simple coalescing-oriented rewrites. After that, I plan to compare estimated warp-level memory transactions before and after transformation on a larger set of kernels.
