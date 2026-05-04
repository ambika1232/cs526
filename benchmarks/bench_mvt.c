// Single-PTX launcher for mvt.cl: x1 = A*y1,  x2 = A^T*y2
// kernel1 is row-stride (coalesced), kernel2 is column-stride (prefetch target).
// Usage: bench_mvt <kernel.ptx> [N]   (N sets square dimension, default 4096)

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
    printf("n=%d  A=%.0f MB\n\n", n, n * (double)n * 4.0 / (1 << 20));

    char *ptx = read_file(argv[1]);
    CUmodule mod; CHECK(cuModuleLoadData(&mod, ptx)); free(ptx);

    CUfunction k1, k2;
    CHECK(cuModuleGetFunction(&k1, mod, "__clang_ocl_kern_imp_mvt_kernel1"));
    CHECK(cuModuleGetFunction(&k2, mod, "__clang_ocl_kern_imp_mvt_kernel2"));

    // a[n*n], x1[n], x2[n], y1[n], y2[n]
    CUdeviceptr d_a, d_x1, d_x2, d_y1, d_y2;
    CHECK(cuMemAlloc(&d_a,  (size_t)n * n * sizeof(float)));
    CHECK(cuMemAlloc(&d_x1, (size_t)n * sizeof(float)));
    CHECK(cuMemAlloc(&d_x2, (size_t)n * sizeof(float)));
    CHECK(cuMemAlloc(&d_y1, (size_t)n * sizeof(float)));
    CHECK(cuMemAlloc(&d_y2, (size_t)n * sizeof(float)));

    float *h = malloc((size_t)n * n * sizeof(float));
    for (int i = 0; i < n * n; i++) h[i] = (float)(i % 256) * 0.01f;
    CHECK(cuMemcpyHtoD(d_a, h, (size_t)n * n * sizeof(float)));
    for (int i = 0; i < n; i++) h[i] = (float)i * 0.5f;
    CHECK(cuMemcpyHtoD(d_y1, h, (size_t)n * sizeof(float)));
    CHECK(cuMemcpyHtoD(d_y2, h, (size_t)n * sizeof(float)));
    free(h);
    CHECK(cuMemsetD32(d_x1, 0, (size_t)n));
    CHECK(cuMemsetD32(d_x2, 0, (size_t)n));

    unsigned g = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    void *args1[] = { &d_a, &d_x1, &d_y1, &n };
    void *args2[] = { &d_a, &d_x2, &d_y2, &n };

    // Warmup
    CHECK(cuLaunchKernel(k1, g, 1, 1, BLOCK_SIZE, 1, 1, 0, NULL, args1, NULL));
    CHECK(cuLaunchKernel(k2, g, 1, 1, BLOCK_SIZE, 1, 1, 0, NULL, args2, NULL));
    CHECK(cuCtxSynchronize());

    CUevent start, stop;
    CHECK(cuEventCreate(&start, CU_EVENT_DEFAULT));
    CHECK(cuEventCreate(&stop,  CU_EVENT_DEFAULT));
    CHECK(cuEventRecord(start, NULL));
    for (int i = 0; i < ITERS; i++) {
        CHECK(cuLaunchKernel(k1, g, 1, 1, BLOCK_SIZE, 1, 1, 0, NULL, args1, NULL));
        CHECK(cuLaunchKernel(k2, g, 1, 1, BLOCK_SIZE, 1, 1, 0, NULL, args2, NULL));
    }
    CHECK(cuEventRecord(stop, NULL));
    CHECK(cuEventSynchronize(stop));

    float ms; CHECK(cuEventElapsedTime(&ms, start, stop));
    printf("avg total time (k1+k2): %.3f ms  (over %d iters)\n", ms / ITERS, ITERS);

    cuEventDestroy(start); cuEventDestroy(stop);
    cuMemFree(d_a); cuMemFree(d_x1); cuMemFree(d_x2);
    cuMemFree(d_y1); cuMemFree(d_y2);
    cuModuleUnload(mod); cuCtxDestroy(ctx);
    return 0;
}
