#include <cuda_runtime.h>
#include <vector>

__global__ void relu_kernel(const float* x, float* out, int n) {
    // out[i] = max(0, x[i])
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = x[idx];
        out[idx] = v > 0.0f ? v : 0.0f;
    }
}

std::vector<float> relu(const std::vector<float>& x) {
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

    // 3. Launch the kernel with enough threads to cover n elements
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    relu_kernel<<<blocks, threads>>>(d_x, d_out, n);

    // 4. Copy the result back to the host
    cudaMemcpy(out.data(), d_out, bytes, cudaMemcpyDeviceToHost);

    // 5. Free device memory
    cudaFree(d_x);
    cudaFree(d_out);

    return out;
}