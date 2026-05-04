// Single-PTX launcher for 3mm.cl: G = (A*B) * (C*D)
// Chains mm3_kernel1 -> mm3_kernel2 -> mm3_kernel3 on the default stream.
// Usage: bench_3mm <kernel.ptx> [N]   (N sets NI=NJ=NK=NL=NM=N, default 4096)

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

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
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "Usage: %s <kernel.ptx> [N]\n", argv[0]);
        return 1;
    }

    int n = (argc >= 3) ? atoi(argv[2]) : 4096;
    int ni = n, nj = n, nk = n, nl = n, nm = n;

    CHECK(cuInit(0));
    CUdevice dev; CHECK(cuDeviceGet(&dev, 0));
    CUcontext ctx; CHECK(cuCtxCreate(&ctx, 0, dev));

    char devname[256]; cuDeviceGetName(devname, sizeof(devname), dev);
    printf("device: %s\n", devname);
    printf("ptx:    %s\n", argv[1]);
    printf("ni=%d nj=%d nk=%d nl=%d nm=%d\n\n", ni, nj, nk, nl, nm);

    char *ptx = read_file(argv[1]);
    CUmodule mod; CHECK(cuModuleLoadData(&mod, ptx)); free(ptx);

    CUfunction k1, k2, k3;
    CHECK(cuModuleGetFunction(&k1, mod, "__clang_ocl_kern_imp_mm3_kernel1"));
    CHECK(cuModuleGetFunction(&k2, mod, "__clang_ocl_kern_imp_mm3_kernel2"));
    CHECK(cuModuleGetFunction(&k3, mod, "__clang_ocl_kern_imp_mm3_kernel3"));

    // A[ni*nk], B[nk*nj], C[nj*nm], D[nm*nl], E[ni*nj], F[nj*nl], G[ni*nl]
    CUdeviceptr d_A, d_B, d_C, d_D, d_E, d_F, d_G;
    CHECK(cuMemAlloc(&d_A, (size_t)ni * nk * sizeof(float)));
    CHECK(cuMemAlloc(&d_B, (size_t)nk * nj * sizeof(float)));
    CHECK(cuMemAlloc(&d_C, (size_t)nj * nm * sizeof(float)));
    CHECK(cuMemAlloc(&d_D, (size_t)nm * nl * sizeof(float)));
    CHECK(cuMemAlloc(&d_E, (size_t)ni * nj * sizeof(float)));
    CHECK(cuMemAlloc(&d_F, (size_t)nj * nl * sizeof(float)));
    CHECK(cuMemAlloc(&d_G, (size_t)ni * nl * sizeof(float)));

    float *h = malloc((size_t)n * n * sizeof(float));
    for (int i = 0; i < ni * nk; i++) h[i] = (float)(i % 256) * 0.01f;
    CHECK(cuMemcpyHtoD(d_A, h, (size_t)ni * nk * sizeof(float)));
    for (int i = 0; i < nk * nj; i++) h[i] = (float)(i % 256) * 0.01f;
    CHECK(cuMemcpyHtoD(d_B, h, (size_t)nk * nj * sizeof(float)));
    for (int i = 0; i < nj * nm; i++) h[i] = (float)(i % 256) * 0.01f;
    CHECK(cuMemcpyHtoD(d_C, h, (size_t)nj * nm * sizeof(float)));
    for (int i = 0; i < nm * nl; i++) h[i] = (float)(i % 256) * 0.01f;
    CHECK(cuMemcpyHtoD(d_D, h, (size_t)nm * nl * sizeof(float)));
    free(h);
    CHECK(cuMemsetD32(d_E, 0, (size_t)ni * nj));
    CHECK(cuMemsetD32(d_F, 0, (size_t)nj * nl));
    CHECK(cuMemsetD32(d_G, 0, (size_t)ni * nl));

    int nj_ = nj;  // alias to pass address for kernel3 arg
    unsigned g1x = (nj + BLOCK_DIM - 1) / BLOCK_DIM;
    unsigned g1y = (ni + BLOCK_DIM - 1) / BLOCK_DIM;
    void *args1[] = { &d_A, &d_B, &d_E, &ni, &nj, &nk };

    unsigned g2x = (nl + BLOCK_DIM - 1) / BLOCK_DIM;
    unsigned g2y = (nj + BLOCK_DIM - 1) / BLOCK_DIM;
    void *args2[] = { &d_C, &d_D, &d_F, &nj, &nl, &nm };

    unsigned g3x = (nl + BLOCK_DIM - 1) / BLOCK_DIM;
    unsigned g3y = (ni + BLOCK_DIM - 1) / BLOCK_DIM;
    void *args3[] = { &d_E, &d_F, &d_G, &ni, &nl, &nj_ };

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
