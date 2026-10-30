#include <cuda_runtime.h>

__global__ void hist_kernel(const int* input, int* hist, int n, int num_bins) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int b = input[i];
        if (b >= 0 && b < num_bins) {
            atomicAdd(&hist[b], 1);
        }
    }
}

void solve(const int* input, int* hist, int n, int num_bins) {
    // Zero the host histogram
    for (int b = 0; b < num_bins; ++b) hist[b] = 0;
    if (n <= 0) return;

    int* d_input = nullptr;
    int* d_hist = nullptr;

    cudaMalloc(&d_input, n * sizeof(int));
    cudaMalloc(&d_hist, num_bins * sizeof(int));

    cudaMemcpy(d_input, input, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_hist, 0, num_bins * sizeof(int));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    hist_kernel<<<blocks, threads>>>(d_input, d_hist, n, num_bins);

    cudaMemcpy(hist, d_hist, num_bins * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_hist);
}