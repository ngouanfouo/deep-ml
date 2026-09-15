#include <cuda_runtime.h>
#include <vector>

__global__ void sum_kernel(const float* x, float* out, int n) {
    __shared__ float sdata[256];

    int tid = threadIdx.x;

    // 1. load x[tid] (or 0) into __shared__ memory, then __syncthreads()
    sdata[tid] = (tid < n) ? x[tid] : 0.0f;
    __syncthreads();

    // 2. tree-reduce: for (s = blockDim.x/2; s > 0; s >>= 1) add sdata[tid+s]
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 3. thread 0 writes out[0]
    if (tid == 0) {
        *out = sdata[0];
    }
}

float array_sum(const std::vector<float>& x) {
    int n = x.size();
    if (n == 0) return 0.0f;

    size_t size_x = n * sizeof(float);
    size_t size_out = sizeof(float);

    float *d_x = nullptr;
    float *d_out = nullptr;
    float h_out = 0.0f;

    // Allocate device memory
    cudaMalloc((void**)&d_x, size_x);
    cudaMalloc((void**)&d_out, size_out);

    // Copy input data from host to device
    cudaMemcpy(d_x, x.data(), size_x, cudaMemcpyHostToDevice);

    // Since n <= 256, a single thread block of 256 threads is sufficient
    int block_size = 256;
    sum_kernel<<<1, block_size>>>(d_x, d_out, n);

    // Copy the resulting sum back to host memory
    cudaMemcpy(&h_out, d_out, size_out, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_x);
    cudaFree(d_out);

    return h_out;
}