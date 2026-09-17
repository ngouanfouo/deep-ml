#include <cuda_runtime.h>
#include <vector>

__global__ void square_kernel(const float* x, float* out, int n) {
    // Grid-stride loop: each thread handles multiple elements if needed.
    int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        out[i] = x[i] * x[i];
    }
}

std::vector<float> square(const std::vector<float>& x) {
    const int n = static_cast<int>(x.size());
    std::vector<float> out(n);
    if (n == 0) return out;

    const size_t bytes = n * sizeof(float);

    // 1. Allocate device memory
    float* d_x = nullptr;
    float* d_out = nullptr;
    cudaMalloc(&d_x, bytes);
    cudaMalloc(&d_out, bytes);

    // 2. Copy input vector to the device
    cudaMemcpy(d_x, x.data(), bytes, cudaMemcpyHostToDevice);

    // 3. Launch a fixed, small grid — the grid-stride loop covers any n.
    const int threads = 256;
    const int max_blocks = 1024;  // cap so tiny arrays don't over-launch
    const int blocks = (n + threads - 1) / threads < max_blocks
                           ? (n + threads - 1) / threads
                           : max_blocks;
    square_kernel<<<blocks, threads>>>(d_x, d_out, n);

    // 4. Copy the result back to the host
    cudaMemcpy(out.data(), d_out, bytes, cudaMemcpyDeviceToHost);

    // 5. Free device memory
    cudaFree(d_x);
    cudaFree(d_out);

    return out;
}