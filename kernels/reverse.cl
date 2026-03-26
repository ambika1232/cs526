__kernel void test(__global float* A, __global float* B, int N) {
    int tid = get_global_id(0);
    B[tid] = A[N - tid - 1];
}
