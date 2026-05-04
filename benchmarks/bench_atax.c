// Single-PTX launcher for atax.cl: y = A^T * (A * x)
// kernel1 computes tmp = A*x, kernel2 computes y = A^T*tmp.
// kernel2 reads tmp written by kernel1 — default stream serializes them.
// Usage: bench_atax <kernel.ptx>

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define NX         4096
#define NY         4096
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
    if (argc != 2) { fprintf(stderr, "Usage: %s <kernel.ptx>\n", argv[0]); return 1; }

    CHECK(cuInit(0));
    CUdevice dev; CHECK(cuDeviceGet(&dev, 0));
    CUcontext ctx; CHECK(cuCtxCreate(&ctx, 0, dev));

    char devname[256]; cuDeviceGetName(devname, sizeof(devname), dev);
    printf("device: %s\n", devname);
    printf("ptx:    %s\n", argv[1]);
    printf("nx=%d ny=%d  A=%.0f MB\n\n", NX, NY, NX * NY * 4.0 / (1 << 20));

    char *ptx = read_file(argv[1]);
    CUmodule mod; CHECK(cuModuleLoadData(&mod, ptx)); free(ptx);

    CUfunction k1, k2;
    CHECK(cuModuleGetFunction(&k1, mod, "__clang_ocl_kern_imp_atax_kernel1"));
    CHECK(cuModuleGetFunction(&k2, mod, "__clang_ocl_kern_imp_atax_kernel2"));

    // A[NX*NY], x[NY], tmp[NX], y[NY]
    CUdeviceptr d_A, d_x, d_tmp, d_y;
    CHECK(cuMemAlloc(&d_A,   (size_t)NX * NY * sizeof(float)));
    CHECK(cuMemAlloc(&d_x,   (size_t)NY * sizeof(float)));
    CHECK(cuMemAlloc(&d_tmp, (size_t)NX * sizeof(float)));
    CHECK(cuMemAlloc(&d_y,   (size_t)NY * sizeof(float)));

    float *h = malloc((size_t)NX * NY * sizeof(float));
    for (int i = 0; i < NX * NY; i++) h[i] = (float)(i % 256) * 0.01f;
    CHECK(cuMemcpyHtoD(d_A, h, (size_t)NX * NY * sizeof(float)));
    for (int i = 0; i < NY; i++) h[i] = (float)i * 0.5f;
    CHECK(cuMemcpyHtoD(d_x, h, (size_t)NY * sizeof(float)));
    free(h);
    // kernel1 does tmp[i] += ..., kernel2 does y[j] += ... — zero once before warmup
    CHECK(cuMemsetD32(d_tmp, 0, (size_t)NX));
    CHECK(cuMemsetD32(d_y,   0, (size_t)NY));

    int nx = NX, ny = NY;
    // kernel1: (A, x, tmp, nx, ny)  grid over NX
    unsigned g1 = (NX + BLOCK_SIZE - 1) / BLOCK_SIZE;
    void *args1[] = { &d_A, &d_x, &d_tmp, &nx, &ny };

    // kernel2: (A, y, tmp, nx, ny)  grid over NY
    unsigned g2 = (NY + BLOCK_SIZE - 1) / BLOCK_SIZE;
    void *args2[] = { &d_A, &d_y, &d_tmp, &nx, &ny };

    // Warmup
    CHECK(cuLaunchKernel(k1, g1, 1, 1, BLOCK_SIZE, 1, 1, 0, NULL, args1, NULL));
    CHECK(cuLaunchKernel(k2, g2, 1, 1, BLOCK_SIZE, 1, 1, 0, NULL, args2, NULL));
    CHECK(cuCtxSynchronize());

    CUevent start, stop;
    CHECK(cuEventCreate(&start, CU_EVENT_DEFAULT));
    CHECK(cuEventCreate(&stop,  CU_EVENT_DEFAULT));
    CHECK(cuEventRecord(start, NULL));
    for (int i = 0; i < ITERS; i++) {
        CHECK(cuLaunchKernel(k1, g1, 1, 1, BLOCK_SIZE, 1, 1, 0, NULL, args1, NULL));
        CHECK(cuLaunchKernel(k2, g2, 1, 1, BLOCK_SIZE, 1, 1, 0, NULL, args2, NULL));
    }
    CHECK(cuEventRecord(stop, NULL));
    CHECK(cuEventSynchronize(stop));

    float ms; CHECK(cuEventElapsedTime(&ms, start, stop));
    printf("avg total time (k1+k2): %.3f ms  (over %d iters)\n", ms / ITERS, ITERS);

    cuEventDestroy(start); cuEventDestroy(stop);
    cuMemFree(d_A); cuMemFree(d_x); cuMemFree(d_tmp); cuMemFree(d_y);
    cuModuleUnload(mod); cuCtxDestroy(ctx);
    return 0;
}
