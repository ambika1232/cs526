// Single-PTX launcher for syrk.cl: C = alpha*A*A' + beta*C
// a[j*ni+k] access (j = get_global_id(0), fast-varying across warp) is
// column-stride across threads — the main prefetch target.
// Usage: bench_syrk <kernel.ptx> [N]   (ni=nj=N, default 4096)

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
    int ni = n, nj = n;

    CHECK(cuInit(0));
    CUdevice dev; CHECK(cuDeviceGet(&dev, 0));
    CUcontext ctx; CHECK(cuCtxCreate(&ctx, 0, dev));

    char devname[256]; cuDeviceGetName(devname, sizeof(devname), dev);
    printf("device: %s\n", devname);
    printf("ptx:    %s\n", argv[1]);
    printf("ni=%d nj=%d  A=%.0f MB  C=%.0f MB\n\n",
           ni, nj,
           nj * (double)ni * 4.0 / (1 << 20),
           nj * (double)nj * 4.0 / (1 << 20));

    char *ptx = read_file(argv[1]);
    CUmodule mod; CHECK(cuModuleLoadData(&mod, ptx)); free(ptx);

    CUfunction kern;
    CHECK(cuModuleGetFunction(&kern, mod, "syrk_kernel"));

    // a[nj*ni], c[nj*nj]
    CUdeviceptr d_a, d_c;
    CHECK(cuMemAlloc(&d_a, (size_t)nj * ni * sizeof(float)));
    CHECK(cuMemAlloc(&d_c, (size_t)nj * nj * sizeof(float)));

    float *h = malloc((size_t)nj * ni * sizeof(float));
    for (int i = 0; i < nj * ni; i++) h[i] = (float)(i % 256) * 0.01f;
    CHECK(cuMemcpyHtoD(d_a, h, (size_t)nj * ni * sizeof(float)));
    free(h);
    h = malloc((size_t)nj * nj * sizeof(float));
    for (int i = 0; i < nj * nj; i++) h[i] = (float)(i % 256) * 0.01f;
    CHECK(cuMemcpyHtoD(d_c, h, (size_t)nj * nj * sizeof(float)));
    free(h);

    float alpha = 1.5f, beta = 1.2f;
    unsigned gx = (nj + BLOCK_DIM - 1) / BLOCK_DIM;
    unsigned gy = (nj + BLOCK_DIM - 1) / BLOCK_DIM;
    void *args[] = { &d_a, &d_c, &alpha, &beta, &ni, &nj };

    // Warmup
    CHECK(cuLaunchKernel(kern, gx, gy, 1, BLOCK_DIM, BLOCK_DIM, 1, 0, NULL, args, NULL));
    CHECK(cuCtxSynchronize());

    CUevent start, stop;
    CHECK(cuEventCreate(&start, CU_EVENT_DEFAULT));
    CHECK(cuEventCreate(&stop,  CU_EVENT_DEFAULT));
    CHECK(cuEventRecord(start, NULL));
    for (int i = 0; i < ITERS; i++)
        CHECK(cuLaunchKernel(kern, gx, gy, 1, BLOCK_DIM, BLOCK_DIM, 1, 0, NULL, args, NULL));
    CHECK(cuEventRecord(stop, NULL));
    CHECK(cuEventSynchronize(stop));

    float ms; CHECK(cuEventElapsedTime(&ms, start, stop));
    printf("avg kernel time: %.3f ms  (over %d iters)\n", ms / ITERS, ITERS);

    cuEventDestroy(start); cuEventDestroy(stop);
    cuMemFree(d_a); cuMemFree(d_c);
    cuModuleUnload(mod); cuCtxDestroy(ctx);
    return 0;
}
