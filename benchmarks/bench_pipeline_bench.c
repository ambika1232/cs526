#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Bench harness for pipeline_bench kernel.
 *
 * Usage: ./bench_pipeline_bench <kernel.ptx> [n]
 *   n = loop iterations per thread (default 32768)
 *       sm_75 (6 MB L2):  n=32768  → 32 MB working set (DRAM bound)
 *       sm_80 (40 MB L2): n=65536  → 64 MB working set
 *       sm_89 (96 MB L2): n=131072 → 128 MB working set
 *
 * Array A: n * 256 floats (one independent lane per thread, stride 256 apart)
 * Grid:    1 block × 256 threads
 */

#define BLOCK_SIZE 256
#define NUM_THREADS 256   /* exactly one block — keeps register pressure low */
#define ITERS       20

#define CHECK(call) do { \
    CUresult _r = (call); \
    if (_r != CUDA_SUCCESS) { \
        const char *_msg; cuGetErrorString(_r, &_msg); \
        fprintf(stderr, "CUDA error at %s:%d — %s\n", __FILE__, __LINE__, _msg); \
        exit(1); \
    } \
} while (0)

static char *read_file(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); long sz = ftell(f); rewind(f);
    char *buf = malloc(sz + 1);
    fread(buf, 1, sz, f); buf[sz] = '\0'; fclose(f);
    return buf;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <kernel.ptx> [n]\n", argv[0]);
        return 1;
    }
    const char *ptx_path = argv[1];
    int n = (argc >= 3) ? atoi(argv[2]) : 32768;
    if (n <= 0) { fprintf(stderr, "n must be positive\n"); return 1; }

    /* Array A: each of the 256 threads reads n elements at stride 256.
     * Thread gid touches A[gid], A[gid+256], A[gid+2*256], ..., A[gid+(n-1)*256].
     * Maximum index = 255 + (n-1)*256 = n*256 - 1.  Allocate n*256 elements. */
    size_t A_elems = (size_t)n * NUM_THREADS;
    size_t A_bytes = A_elems * sizeof(float);

    CHECK(cuInit(0));
    CUdevice dev; CHECK(cuDeviceGet(&dev, 0));
    CUcontext ctx; CHECK(cuCtxCreate(&ctx, 0, dev));

    char name[256]; cuDeviceGetName(name, sizeof(name), dev);
    printf("device:       %s\n", name);
    printf("ptx:          %s\n", ptx_path);
    printf("n (iters):    %d\n", n);
    printf("threads:      %d\n", NUM_THREADS);
    printf("working set:  %.1f MB\n\n", A_bytes / 1e6);

    char *ptx = read_file(ptx_path);
    CUmodule mod; CHECK(cuModuleLoadData(&mod, ptx)); free(ptx);

    CUfunction fn;
    CHECK(cuModuleGetFunction(&fn, mod, "pipeline_bench"));

    CUdeviceptr d_A, d_out;
    CHECK(cuMemAlloc(&d_A,   A_bytes));
    CHECK(cuMemAlloc(&d_out, NUM_THREADS * sizeof(float)));

    /* Fill A with small floats so the sum doesn't overflow */
    float *h_A = malloc(A_bytes);
    for (size_t i = 0; i < A_elems; i++) h_A[i] = (float)(i % 1000) * 0.001f;
    CHECK(cuMemcpyHtoD(d_A, h_A, A_bytes));
    free(h_A);

    void *args[] = { &d_A, &d_out, &n };

    /* Warmup */
    CHECK(cuLaunchKernel(fn, 1, 1, 1, BLOCK_SIZE, 1, 1, 0, NULL, args, NULL));
    CHECK(cuCtxSynchronize());

    /* Timed runs */
    CUevent t0, t1;
    CHECK(cuEventCreate(&t0, CU_EVENT_DEFAULT));
    CHECK(cuEventCreate(&t1, CU_EVENT_DEFAULT));
    CHECK(cuEventRecord(t0, NULL));
    for (int i = 0; i < ITERS; i++)
        CHECK(cuLaunchKernel(fn, 1, 1, 1, BLOCK_SIZE, 1, 1, 0, NULL, args, NULL));
    CHECK(cuEventRecord(t1, NULL));
    CHECK(cuEventSynchronize(t1));

    float ms; CHECK(cuEventElapsedTime(&ms, t0, t1));
    printf("avg kernel time: %.3f ms  (over %d iters)\n", ms / ITERS, ITERS);

    cuEventDestroy(t0); cuEventDestroy(t1);
    cuMemFree(d_A); cuMemFree(d_out);
    cuModuleUnload(mod); cuCtxDestroy(ctx);
    return 0;
}
