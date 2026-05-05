__kernel void test_prefetch(__global float *A,
                            __global float *B,
                            int N) {
    int tid = get_global_id(0);
    float sum = 0.0f;
    
    for (int i = tid; i < N; i += 64) {
        sum += A[i];
    }

    B[tid] = sum;
}