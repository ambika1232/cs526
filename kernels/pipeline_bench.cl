/**
 * pipeline_bench.cl — synthetic kernel designed to isolate pipeline-pass benefit.
 *
 * Access pattern: A[gid + k * 256]
 *   - stride 256 is a compile-time literal → SCEV sees {gid, +, 1024} (constant step)
 *   - pipeline pass selects this load (not skipped as symbolic)
 *   - hint pass also inserts prefetch.global.L2 D iterations ahead
 *
 * Unlike PolyBench kernels, the bottleneck load here IS the constant-stride one.
 * There is no small auxiliary vector stealing the pass's attention.
 *
 * Working set: 256 threads × n iterations × 4 bytes = 1024n bytes
 *   sm_75  (6 MB L2): n ≥  6144 forces DRAM traffic; default n = 32768 (32 MB)
 *   sm_80  (40 MB L2): n ≥ 40960; use n = 65536 (64 MB)
 *   sm_89  (96 MB L2): n ≥ 98304; use n = 131072 (128 MB)
 *
 * Expected behaviour:
 *   hint pass:     significant speedup (~20-30%) — D=16 provides enough advance
 *   pipeline pass: smaller speedup — 1-step lookahead gives 35 cycles cover
 *                  vs 350 cycle DRAM latency on sm_75; more visible when SM
 *                  occupancy is low (fewer warps to do latency hiding naturally)
 *
 * Array A must have at least (n * 256) elements allocated by the host.
 * Array out must have at least 256 elements.
 *
 * Build with run_gpu_benchmark.sh:
 *   bash scripts/run_gpu_benchmark.sh kernels/pipeline_bench.cl sm_75
 */

typedef float DATA_TYPE;

__kernel void pipeline_bench(
    __global DATA_TYPE *A,
    __global DATA_TYPE *out,
    int n
) {
    int gid = get_global_id(0);

    DATA_TYPE sum = 0.0f;

    for (int k = 0; k < n; k++) {
        sum += A[gid + k * 256];
    }

    out[gid] = sum;
}
