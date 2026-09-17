#include <cuda_runtime.h>
#include <vector>

__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n) {
    // Each thread handles exactly one element.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

std::vector<float> vector_add(const std::vector<float>& a, const std::vector<float>& b) {
    const int n = static_cast<int>(a.size());
    std::vector<float> c(n);
    if (n == 0) return c;

    const size_t bytes = n * sizeof(float);

    // 1. Allocate device memory
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // 2. Copy inputs to the device
    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    // 3. Launch the kernel with enough threads to cover n elements
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    vector_add_kernel<<<blocks, threads>>>(d_a, d_b, d_c, n);

    // 4. Copy the result back to the host
    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    // 5. Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}