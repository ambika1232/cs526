__kernel void bounded_strided(__global float *A,
                             __global float *B,
                             int N)
{
  int tid = get_global_id(0);
  if (4 * tid < N)
    B[tid] = A[4 * tid];
}