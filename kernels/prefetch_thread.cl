__kernel void stencil1d(__global float *In,
                        __global float *Out,
                        int N) {
    int tid = get_global_id(0);
    if (tid > 0 && tid < N - 1) {
        float x0 = In[tid - 1];
        float x1 = In[tid];
        float x2 = In[tid + 1];
        Out[tid] = x0 + x1 + x2;
    }
}