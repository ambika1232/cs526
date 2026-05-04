// Single-PTX launcher for 2mm.cl: D = beta*D + (alpha*A*B) * C
// Chains mm2_kernel1 -> mm2_kernel2 on the default stream (serialized automatically).
// Usage: bench_2mm <kernel.ptx>

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define NI         1024
#define NJ         1024
#define NK         1024
#define NL         1024
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
    printf("ni=%d nj=%d nk=%d nl=%d\n\n", NI, NJ, NK, NL);

    char *ptx = read_file(argv[1]);
    CUmodule mod; CHECK(cuModuleLoadData(&mod, ptx)); free(ptx);

    CUfunction k1, k2;
    CHECK(cuModuleGetFunction(&k1, mod, "__clang_ocl_kern_imp_mm2_kernel1"));
    CHECK(cuModuleGetFunction(&k2, mod, "__clang_ocl_kern_imp_mm2_kernel2"));

    // tmp[NI*NJ], A[NI*NK], B[NK*NJ], C[NJ*NL], D[NI*NL]
    CUdeviceptr d_tmp, d_A, d_B, d_C, d_D;
    CHECK(cuMemAlloc(&d_tmp, (size_t)NI * NJ * sizeof(float)));
    CHECK(cuMemAlloc(&d_A,   (size_t)NI * NK * sizeof(float)));
    CHECK(cuMemAlloc(&d_B,   (size_t)NK * NJ * sizeof(float)));
    CHECK(cuMemAlloc(&d_C,   (size_t)NJ * NL * sizeof(float)));
    CHECK(cuMemAlloc(&d_D,   (size_t)NI * NL * sizeof(float)));

    float *h = malloc((size_t)NI * NJ * sizeof(float));
    for (int i = 0; i < NI * NK; i++) h[i] = (float)(i % 256) * 0.01f;
    CHECK(cuMemcpyHtoD(d_A, h, (size_t)NI * NK * sizeof(float)));
    for (int i = 0; i < NK * NJ; i++) h[i] = (float)(i % 256) * 0.01f;
    CHECK(cuMemcpyHtoD(d_B, h, (size_t)NK * NJ * sizeof(float)));
    for (int i = 0; i < NJ * NL; i++) h[i] = (float)(i % 256) * 0.01f;
    CHECK(cuMemcpyHtoD(d_C, h, (size_t)NJ * NL * sizeof(float)));
    for (int i = 0; i < NI * NL; i++) h[i] = 1.0f;
    CHECK(cuMemcpyHtoD(d_D, h, (size_t)NI * NL * sizeof(float)));
    free(h);
    CHECK(cuMemsetD32(d_tmp, 0, (size_t)NI * NJ));

    float alpha = 1.5f, beta = 1.2f;
    int ni = NI, nj = NJ, nk = NK, nl = NL;

    // kernel1: (tmp, A, B, ni, nj, nk, nl, alpha, beta)  grid covers NI x NJ
    unsigned g1x = (NJ + BLOCK_DIM - 1) / BLOCK_DIM;
    unsigned g1y = (NI + BLOCK_DIM - 1) / BLOCK_DIM;
    void *args1[] = { &d_tmp, &d_A, &d_B, &ni, &nj, &nk, &nl, &alpha, &beta };

    // kernel2: (tmp, C, D, ni, nj, nk, nl, alpha, beta)  grid covers NI x NL
    unsigned g2x = (NL + BLOCK_DIM - 1) / BLOCK_DIM;
    unsigned g2y = (NI + BLOCK_DIM - 1) / BLOCK_DIM;
    void *args2[] = { &d_tmp, &d_C, &d_D, &ni, &nj, &nk, &nl, &alpha, &beta };

    // Warmup
    CHECK(cuLaunchKernel(k1, g1x, g1y, 1, BLOCK_DIM, BLOCK_DIM, 1, 0, NULL, args1, NULL));
    CHECK(cuLaunchKernel(k2, g2x, g2y, 1, BLOCK_DIM, BLOCK_DIM, 1, 0, NULL, args2, NULL));
    CHECK(cuCtxSynchronize());

    CUevent start, stop;
    CHECK(cuEventCreate(&start, CU_EVENT_DEFAULT));
    CHECK(cuEventCreate(&stop,  CU_EVENT_DEFAULT));
    CHECK(cuEventRecord(start, NULL));
    for (int i = 0; i < ITERS; i++) {
        CHECK(cuLaunchKernel(k1, g1x, g1y, 1, BLOCK_DIM, BLOCK_DIM, 1, 0, NULL, args1, NULL));
        CHECK(cuLaunchKernel(k2, g2x, g2y, 1, BLOCK_DIM, BLOCK_DIM, 1, 0, NULL, args2, NULL));
    }
    CHECK(cuEventRecord(stop, NULL));
    CHECK(cuEventSynchronize(stop));

    float ms; CHECK(cuEventElapsedTime(&ms, start, stop));
    printf("avg total time (k1+k2): %.3f ms  (over %d iters)\n", ms / ITERS, ITERS);

    cuEventDestroy(start); cuEventDestroy(stop);
    cuMemFree(d_tmp); cuMemFree(d_A); cuMemFree(d_B); cuMemFree(d_C); cuMemFree(d_D);
    cuModuleUnload(mod); cuCtxDestroy(ctx);
    return 0;
}
