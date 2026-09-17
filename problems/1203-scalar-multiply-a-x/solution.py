#include <cuda_runtime.h>
#include <vector>

__global__ void scale_kernel(const float* x, float a, float* y, int n) {
    // y[i] = a * x[i]
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = a * x[idx];
    }
}

std::vector<float> scalar_multiply(const std::vector<float>& x, float a) {
    const int n = static_cast<int>(x.size());
    std::vector<float> y(n);
    if (n == 0) return y;

    const size_t bytes = n * sizeof(float);

    // 1. Allocate device memory
    float* d_x = nullptr;
    float* d_y = nullptr;
    cudaMalloc(&d_x, bytes);
    cudaMalloc(&d_y, bytes);

    // 2. Copy input vector to the device
    cudaMemcpy(d_x, x.data(), bytes, cudaMemcpyHostToDevice);

    // 3. Launch the kernel, passing the scalar `a` by value
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    scale_kernel<<<blocks, threads>>>(d_x, a, d_y, n);

    // 4. Copy the result back to the host
    cudaMemcpy(y.data(), d_y, bytes, cudaMemcpyDeviceToHost);

    // 5. Free device memory
    cudaFree(d_x);
    cudaFree(d_y);

    return y;
}