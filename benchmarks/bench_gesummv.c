// Single-PTX launcher for gesummv.cl: y = alpha*(A*x) + beta*(B*x)
// Single kernel; both A and B are row-stride (coalesced), still benefits from
// L2 hint prefetch when working set (2*N^2*4 bytes) exceeds L2.
// Usage: bench_gesummv <kernel.ptx> [N]   (default 4096)

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 256
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

    CHECK(cuInit(0));
    CUdevice dev; CHECK(cuDeviceGet(&dev, 0));
    CUcontext ctx; CHECK(cuCtxCreate(&ctx, 0, dev));

    char devname[256]; cuDeviceGetName(devname, sizeof(devname), dev);
    printf("device: %s\n", devname);
    printf("ptx:    %s\n", argv[1]);
    printf("n=%d  A+B=%.0f MB\n\n", n, 2.0 * n * (double)n * 4.0 / (1 << 20));

    char *ptx = read_file(argv[1]);
    CUmodule mod; CHECK(cuModuleLoadData(&mod, ptx)); free(ptx);

    CUfunction kern;
    CHECK(cuModuleGetFunction(&kern, mod, "gesummv_kernel"));

    // a[n*n], b[n*n], x[n], y[n], tmp[n]
    CUdeviceptr d_a, d_b, d_x, d_y, d_tmp;
    CHECK(cuMemAlloc(&d_a,   (size_t)n * n * sizeof(float)));
    CHECK(cuMemAlloc(&d_b,   (size_t)n * n * sizeof(float)));
    CHECK(cuMemAlloc(&d_x,   (size_t)n * sizeof(float)));
    CHECK(cuMemAlloc(&d_y,   (size_t)n * sizeof(float)));
    CHECK(cuMemAlloc(&d_tmp, (size_t)n * sizeof(float)));

    float *h = malloc((size_t)n * n * sizeof(float));
    for (int i = 0; i < n * n; i++) h[i] = (float)(i % 256) * 0.01f;
    CHECK(cuMemcpyHtoD(d_a, h, (size_t)n * n * sizeof(float)));
    CHECK(cuMemcpyHtoD(d_b, h, (size_t)n * n * sizeof(float)));
    for (int i = 0; i < n; i++) h[i] = (float)i * 0.5f;
    CHECK(cuMemcpyHtoD(d_x, h, (size_t)n * sizeof(float)));
    free(h);
    CHECK(cuMemsetD32(d_y,   0, (size_t)n));
    CHECK(cuMemsetD32(d_tmp, 0, (size_t)n));

    float alpha = 1.5f, beta = 1.2f;
    unsigned g = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    void *args[] = { &d_a, &d_b, &d_x, &d_y, &d_tmp, &alpha, &beta, &n };

    // Warmup
    CHECK(cuLaunchKernel(kern, g, 1, 1, BLOCK_SIZE, 1, 1, 0, NULL, args, NULL));
    CHECK(cuCtxSynchronize());

    CUevent start, stop;
    CHECK(cuEventCreate(&start, CU_EVENT_DEFAULT));
    CHECK(cuEventCreate(&stop,  CU_EVENT_DEFAULT));
    CHECK(cuEventRecord(start, NULL));
    for (int i = 0; i < ITERS; i++)
        CHECK(cuLaunchKernel(kern, g, 1, 1, BLOCK_SIZE, 1, 1, 0, NULL, args, NULL));
    CHECK(cuEventRecord(stop, NULL));
    CHECK(cuEventSynchronize(stop));

    float ms; CHECK(cuEventElapsedTime(&ms, start, stop));
    printf("avg kernel time: %.3f ms  (over %d iters)\n", ms / ITERS, ITERS);

    cuEventDestroy(start); cuEventDestroy(stop);
    cuMemFree(d_a); cuMemFree(d_b); cuMemFree(d_x);
    cuMemFree(d_y); cuMemFree(d_tmp);
    cuModuleUnload(mod); cuCtxDestroy(ctx);
    return 0;
}
