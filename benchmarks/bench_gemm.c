// Single-PTX launcher for gemm.cl: C = alpha*A*B + beta*C
// Usage: bench_gemm <kernel.ptx> [N]   (N sets NI=NJ=NK=N, default 4096)
// nsys: nsys profile -o report ./bench_gemm kernel.ptx [N]

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
    int ni = n, nj = n, nk = n;

    CHECK(cuInit(0));
    CUdevice dev; CHECK(cuDeviceGet(&dev, 0));
    CUcontext ctx; CHECK(cuCtxCreate(&ctx, 0, dev));

    char devname[256]; cuDeviceGetName(devname, sizeof(devname), dev);
    printf("device: %s\n", devname);
    printf("ptx:    %s\n", argv[1]);
    printf("ni=%d nj=%d nk=%d (%.0f MB/matrix)\n\n",
           ni, nj, nk, ni * (double)nj * 4.0 / (1 << 20));

    char *ptx = read_file(argv[1]);
    CUmodule mod; CHECK(cuModuleLoadData(&mod, ptx)); free(ptx);

    CUfunction fn;
    CHECK(cuModuleGetFunction(&fn, mod, "__clang_ocl_kern_imp_gemm"));

    CUdeviceptr d_a, d_b, d_c;
    CHECK(cuMemAlloc(&d_a, (size_t)ni * nk * sizeof(float)));
    CHECK(cuMemAlloc(&d_b, (size_t)nk * nj * sizeof(float)));
    CHECK(cuMemAlloc(&d_c, (size_t)ni * nj * sizeof(float)));

    float *h = malloc((size_t)n * n * sizeof(float));
    for (int i = 0; i < ni * nk; i++) h[i] = (float)(i % 256) * 0.01f;
    CHECK(cuMemcpyHtoD(d_a, h, (size_t)ni * nk * sizeof(float)));
    for (int i = 0; i < nk * nj; i++) h[i] = (float)(i % 256) * 0.01f;
    CHECK(cuMemcpyHtoD(d_b, h, (size_t)nk * nj * sizeof(float)));
    for (int i = 0; i < ni * nj; i++) h[i] = 1.0f;
    CHECK(cuMemcpyHtoD(d_c, h, (size_t)ni * nj * sizeof(float)));
    free(h);

    float alpha = 1.5f, beta = 1.2f;
    unsigned gridX = (nj + BLOCK_DIM - 1) / BLOCK_DIM;
    unsigned gridY = (ni + BLOCK_DIM - 1) / BLOCK_DIM;
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
