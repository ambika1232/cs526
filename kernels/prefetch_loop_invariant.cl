__kernel void test_no_prefetch_invariant(__global float *A,
                                         __global float *B,
                                         int N) {
    int tid = get_global_id(0);
    float sum = 0.0f;

    for (int i = 0; i < N; i += 64) {
        sum += A[tid];
    }

    B[tid] = sum;
}