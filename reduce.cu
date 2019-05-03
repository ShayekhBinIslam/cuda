__global__ void global_reduce_kernel(float * d_out, float * d_in)
{
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int tid = threadIdx.x;

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            d_in[myId] += d_in[myId + s];
        }
        __syncthreads();
    }
    // thread 0 outputs
    if (tid == 0) {
        d_out[blockIdx.x] = d_in[myId];
    }
}

__global__ void shared_reduce_kernel(float * d_out, float * d_in) {
    // s data allcoated in kernel call's third argument
    extern __shared__ float sdata[];

    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int tid = threadIdx.x;

    // load shared from global
    sdata[tid] = d_in[myId];
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[myId] += d_in[myId + s];
        }
        __syncthreads();
    }
    // thread 0 outputs
    if (tid == 0) {
        d_out[blockIdx.x] = sdata[0];
    }
}
