#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

// Single-PTX launcher — run under nsys to profile.
// Usage: bench_prefetch <kernel.ptx>
// nsys: nsys profile -o report ./bench_prefetch kernel.ptx

#define N          (1 << 24)   // 16M floats (~64 MB per array)
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
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    rewind(f);
    char *buf = malloc(sz + 1);
    fread(buf, 1, sz, f);
    buf[sz] = '\0';
    fclose(f);
    return buf;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <kernel.ptx>\n", argv[0]);
        return 1;
    }

    CHECK(cuInit(0));
    CUdevice dev; CHECK(cuDeviceGet(&dev, 0));
    CUcontext ctx; CHECK(cuCtxCreate(&ctx, 0, dev));

    char name[256];
    cuDeviceGetName(name, sizeof(name), dev);
    printf("device: %s\n", name);
    printf("ptx:    %s\n", argv[1]);
    printf("n:      %d elements (%.0f MB/array)\n\n", N, N * 4.0 / (1 << 20));

    char *ptx = read_file(argv[1]);
    CUmodule mod;
    CHECK(cuModuleLoadData(&mod, ptx));
    free(ptx);

    CUfunction fn;
    CHECK(cuModuleGetFunction(&fn, mod, "test_prefetch"));

    CUdeviceptr d_A, d_B;
    CHECK(cuMemAlloc(&d_A, (size_t)N * sizeof(float)));
    CHECK(cuMemAlloc(&d_B, (size_t)N * sizeof(float)));

    float *h_A = malloc((size_t)N * sizeof(float));
    for (int i = 0; i < N; i++) h_A[i] = (float)i * 0.001f;
    CHECK(cuMemcpyHtoD(d_A, h_A, (size_t)N * sizeof(float)));
    free(h_A);

    int n = N;
    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    void *args[] = { &d_A, &d_B, &n };

    // Warmup — not captured by nsys NVTX range
    CHECK(cuLaunchKernel(fn, grid, 1, 1, BLOCK_SIZE, 1, 1, 0, NULL, args, NULL));
    CHECK(cuCtxSynchronize());

    // Timed runs — nsys sees all ITERS kernel launches
    CUevent start, stop;
    CHECK(cuEventCreate(&start, CU_EVENT_DEFAULT));
    CHECK(cuEventCreate(&stop,  CU_EVENT_DEFAULT));
    CHECK(cuEventRecord(start, NULL));
    for (int i = 0; i < ITERS; i++)
        CHECK(cuLaunchKernel(fn, grid, 1, 1, BLOCK_SIZE, 1, 1, 0, NULL, args, NULL));
    CHECK(cuEventRecord(stop, NULL));
    CHECK(cuEventSynchronize(stop));

    float ms;
    CHECK(cuEventElapsedTime(&ms, start, stop));
    printf("avg kernel time: %.3f ms  (over %d iters)\n", ms / ITERS, ITERS);

    cuEventDestroy(start);
    cuEventDestroy(stop);
    cuMemFree(d_A);
    cuMemFree(d_B);
    cuModuleUnload(mod);
    cuCtxDestroy(ctx);
    return 0;
}
