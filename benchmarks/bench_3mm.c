// Single-PTX launcher for 3mm.cl: G = (A*B) * (C*D)
// Chains mm3_kernel1 -> mm3_kernel2 -> mm3_kernel3 on the default stream.
// Usage: bench_3mm <kernel.ptx>

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define NI         1024
#define NJ         1024
#define NK         1024
#define NL         1024
#define NM         1024
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
    printf("ni=%d nj=%d nk=%d nl=%d nm=%d\n\n", NI, NJ, NK, NL, NM);

    char *ptx = read_file(argv[1]);
    CUmodule mod; CHECK(cuModuleLoadData(&mod, ptx)); free(ptx);

    CUfunction k1, k2, k3;
    CHECK(cuModuleGetFunction(&k1, mod, "__clang_ocl_kern_imp_mm3_kernel1"));
    CHECK(cuModuleGetFunction(&k2, mod, "__clang_ocl_kern_imp_mm3_kernel2"));
    CHECK(cuModuleGetFunction(&k3, mod, "__clang_ocl_kern_imp_mm3_kernel3"));

    // A[NI*NK], B[NK*NJ], C[NJ*NM], D[NM*NL], E[NI*NJ], F[NJ*NL], G[NI*NL]
    CUdeviceptr d_A, d_B, d_C, d_D, d_E, d_F, d_G;
    CHECK(cuMemAlloc(&d_A, (size_t)NI * NK * sizeof(float)));
    CHECK(cuMemAlloc(&d_B, (size_t)NK * NJ * sizeof(float)));
    CHECK(cuMemAlloc(&d_C, (size_t)NJ * NM * sizeof(float)));
    CHECK(cuMemAlloc(&d_D, (size_t)NM * NL * sizeof(float)));
    CHECK(cuMemAlloc(&d_E, (size_t)NI * NJ * sizeof(float)));
    CHECK(cuMemAlloc(&d_F, (size_t)NJ * NL * sizeof(float)));
    CHECK(cuMemAlloc(&d_G, (size_t)NI * NL * sizeof(float)));

    float *h = malloc((size_t)NI * NJ * sizeof(float));
    for (int i = 0; i < NI * NK; i++) h[i] = (float)(i % 256) * 0.01f;
    CHECK(cuMemcpyHtoD(d_A, h, (size_t)NI * NK * sizeof(float)));
    for (int i = 0; i < NK * NJ; i++) h[i] = (float)(i % 256) * 0.01f;
    CHECK(cuMemcpyHtoD(d_B, h, (size_t)NK * NJ * sizeof(float)));
    for (int i = 0; i < NJ * NM; i++) h[i] = (float)(i % 256) * 0.01f;
    CHECK(cuMemcpyHtoD(d_C, h, (size_t)NJ * NM * sizeof(float)));
    for (int i = 0; i < NM * NL; i++) h[i] = (float)(i % 256) * 0.01f;
    CHECK(cuMemcpyHtoD(d_D, h, (size_t)NM * NL * sizeof(float)));
    free(h);
    // E, F, G are outputs; kernels initialize them to 0 before accumulating
    CHECK(cuMemsetD32(d_E, 0, (size_t)NI * NJ));
    CHECK(cuMemsetD32(d_F, 0, (size_t)NJ * NL));
    CHECK(cuMemsetD32(d_G, 0, (size_t)NI * NL));

    int ni = NI, nj = NJ, nk = NK, nl = NL, nm = NM;

    // kernel1: (A, B, E, ni, nj, nk)   E = A*B   grid NI x NJ
    unsigned g1x = (NJ + BLOCK_DIM - 1) / BLOCK_DIM;
    unsigned g1y = (NI + BLOCK_DIM - 1) / BLOCK_DIM;
    void *args1[] = { &d_A, &d_B, &d_E, &ni, &nj, &nk };

    // kernel2: (C, D, F, nj, nl, nm)   F = C*D   grid NJ x NL
    unsigned g2x = (NL + BLOCK_DIM - 1) / BLOCK_DIM;
    unsigned g2y = (NJ + BLOCK_DIM - 1) / BLOCK_DIM;
    void *args2[] = { &d_C, &d_D, &d_F, &nj, &nl, &nm };

    // kernel3: (E, F, G, ni, nl, nj)   G = E*F   grid NI x NL
    unsigned g3x = (NL + BLOCK_DIM - 1) / BLOCK_DIM;
    unsigned g3y = (NI + BLOCK_DIM - 1) / BLOCK_DIM;
    void *args3[] = { &d_E, &d_F, &d_G, &ni, &nl, &nj };

    // Warmup
    CHECK(cuLaunchKernel(k1, g1x, g1y, 1, BLOCK_DIM, BLOCK_DIM, 1, 0, NULL, args1, NULL));
    CHECK(cuLaunchKernel(k2, g2x, g2y, 1, BLOCK_DIM, BLOCK_DIM, 1, 0, NULL, args2, NULL));
    CHECK(cuLaunchKernel(k3, g3x, g3y, 1, BLOCK_DIM, BLOCK_DIM, 1, 0, NULL, args3, NULL));
    CHECK(cuCtxSynchronize());

    CUevent start, stop;
    CHECK(cuEventCreate(&start, CU_EVENT_DEFAULT));
    CHECK(cuEventCreate(&stop,  CU_EVENT_DEFAULT));
    CHECK(cuEventRecord(start, NULL));
    for (int i = 0; i < ITERS; i++) {
        CHECK(cuLaunchKernel(k1, g1x, g1y, 1, BLOCK_DIM, BLOCK_DIM, 1, 0, NULL, args1, NULL));
        CHECK(cuLaunchKernel(k2, g2x, g2y, 1, BLOCK_DIM, BLOCK_DIM, 1, 0, NULL, args2, NULL));
        CHECK(cuLaunchKernel(k3, g3x, g3y, 1, BLOCK_DIM, BLOCK_DIM, 1, 0, NULL, args3, NULL));
    }
    CHECK(cuEventRecord(stop, NULL));
    CHECK(cuEventSynchronize(stop));

    float ms; CHECK(cuEventElapsedTime(&ms, start, stop));
    printf("avg total time (k1+k2+k3): %.3f ms  (over %d iters)\n", ms / ITERS, ITERS);

    cuEventDestroy(start); cuEventDestroy(stop);
    cuMemFree(d_A); cuMemFree(d_B); cuMemFree(d_C); cuMemFree(d_D);
    cuMemFree(d_E); cuMemFree(d_F); cuMemFree(d_G);
    cuModuleUnload(mod); cuCtxDestroy(ctx);
    return 0;
}
