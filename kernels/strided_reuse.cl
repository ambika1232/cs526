__kernel void test(__global float* A, __global float* B) {
    int tid = get_global_id(0);
    B[tid] = A[tid * 4];
}
