// Single-PTX launcher for gemm.cl: C = alpha*A*B + beta*C
// Usage: bench_gemm <kernel.ptx>
// nsys: nsys profile -o report ./bench_gemm kernel.ptx

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define NI         1024
#define NJ         1024
#define NK         1024
#define BLOCK_DIM  16
#define ITERS      20

#define CHECK(call) do { \
    CUresult _r = (call); \
    if (_r != CUDA_SUCCESS) { \
        const char *_msg; \
        cuGetErrorString(_r, &_msg); \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, _msg); \
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
    if (argc != 2) { fprintf(stderr, "Usage: %s <kernel.ptx>\n", argv[0]); return 1; }

    CHECK(cuInit(0));
    CUdevice dev; CHECK(cuDeviceGet(&dev, 0));
    CUcontext ctx; CHECK(cuCtxCreate(&ctx, 0, dev));

    char devname[256]; cuDeviceGetName(devname, sizeof(devname), dev);
    printf("device: %s\n", devname);
    printf("ptx:    %s\n", argv[1]);
    printf("ni=%d nj=%d nk=%d (%.0f MB/matrix)\n\n",
           NI, NJ, NK, NI * NJ * 4.0 / (1 << 20));

    char *ptx = read_file(argv[1]);
    CUmodule mod; CHECK(cuModuleLoadData(&mod, ptx)); free(ptx);

    CUfunction fn;
    CHECK(cuModuleGetFunction(&fn, mod, "__clang_ocl_kern_imp_gemm"));

    CUdeviceptr d_a, d_b, d_c;
    CHECK(cuMemAlloc(&d_a, (size_t)NI * NK * sizeof(float)));
    CHECK(cuMemAlloc(&d_b, (size_t)NK * NJ * sizeof(float)));
    CHECK(cuMemAlloc(&d_c, (size_t)NI * NJ * sizeof(float)));

    // NI*NJ is the largest allocation — use it for all init passes
    float *h = malloc((size_t)NI * NJ * sizeof(float));
    for (int i = 0; i < NI * NK; i++) h[i] = (float)(i % 256) * 0.01f;
    CHECK(cuMemcpyHtoD(d_a, h, (size_t)NI * NK * sizeof(float)));
    for (int i = 0; i < NK * NJ; i++) h[i] = (float)(i % 256) * 0.01f;
    CHECK(cuMemcpyHtoD(d_b, h, (size_t)NK * NJ * sizeof(float)));
    for (int i = 0; i < NI * NJ; i++) h[i] = 1.0f;
    CHECK(cuMemcpyHtoD(d_c, h, (size_t)NI * NJ * sizeof(float)));
    free(h);

    float alpha = 1.5f, beta = 1.2f;
    int ni = NI, nj = NJ, nk = NK;
    // get_global_id(0)=j (columns), get_global_id(1)=i (rows)
    unsigned gridX = (NJ + BLOCK_DIM - 1) / BLOCK_DIM;
    unsigned gridY = (NI + BLOCK_DIM - 1) / BLOCK_DIM;
    void *args[] = { &d_a, &d_b, &d_c, &alpha, &beta, &ni, &nj, &nk };

    CHECK(cuLaunchKernel(fn, gridX, gridY, 1, BLOCK_DIM, BLOCK_DIM, 1, 0, NULL, args, NULL));
    CHECK(cuCtxSynchronize());

    CUevent start, stop;
    CHECK(cuEventCreate(&start, CU_EVENT_DEFAULT));
    CHECK(cuEventCreate(&stop,  CU_EVENT_DEFAULT));
    CHECK(cuEventRecord(start, NULL));
    for (int i = 0; i < ITERS; i++)
        CHECK(cuLaunchKernel(fn, gridX, gridY, 1, BLOCK_DIM, BLOCK_DIM, 1, 0, NULL, args, NULL));
    CHECK(cuEventRecord(stop, NULL));
    CHECK(cuEventSynchronize(stop));

    float ms; CHECK(cuEventElapsedTime(&ms, start, stop));
    printf("avg kernel time: %.3f ms  (over %d iters)\n", ms / ITERS, ITERS);

    cuEventDestroy(start); cuEventDestroy(stop);
    cuMemFree(d_a); cuMemFree(d_b); cuMemFree(d_c);
    cuModuleUnload(mod); cuCtxDestroy(ctx);
    return 0;
}
