#include <cuda_runtime.h>

__global__ void reduce_kernel(const float* x, float* out, int n) {
    float sum = 0.0f;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) sum = x[i];

    // Intra-warp reduction via shuffle
    for (int s = 16; s > 0; s >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, s);
    }

    // Stage warp totals in shared memory
    __shared__ float warp_sums[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;

    if (lane == 0) warp_sums[wid] = sum;
    __syncthreads();

    // First warp combines the warp totals
    int num_warps = (blockDim.x + 31) >> 5;
    if (wid == 0) {
        float v = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        for (int s = 16; s > 0; s >>= 1) {
            v += __shfl_down_sync(0xffffffff, v, s);
        }
        if (lane == 0) atomicAdd(out, v);
    }
}

void solve(const float* x, float* out, int n) {
    float* d_x = nullptr;
    float* d_out = nullptr;

    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_out, sizeof(float));

    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, sizeof(float));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    reduce_kernel<<<blocks, threads>>>(d_x, d_out, n);

    cudaMemcpy(out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_out);
}