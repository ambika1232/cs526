__kernel void reject_plus_one(__global float *A, __global float *B) {
  int tid = get_global_id(0);
  B[tid] = A[4 * tid + 1];
}

__kernel void reject_minus_one(__global float *A, __global float *B) {
  int tid = get_global_id(0);
  B[tid] = A[4 * tid - 1];
}

__kernel void reject_coalesced(__global float *A, __global float *B) {
  int tid = get_global_id(0);
  B[tid] = A[tid];
}

__kernel void accept_clean_strided(__global float *A, __global float *B) {
  int tid = get_global_id(0);
  B[tid] = A[4 * tid];
}