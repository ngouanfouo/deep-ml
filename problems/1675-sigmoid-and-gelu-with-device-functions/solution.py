#include <cuda_runtime.h>
#include <math.h>

__device__ float sigmoid_dev(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ float gelu_dev(float x) {
    float c = sqrtf(2.0f / 3.14159265358979323846f);
    float inner = c * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

__global__ void activations_kernel(const float* x, float* sig, float* gelu, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        sig[i] = sigmoid_dev(v);
        gelu[i] = gelu_dev(v);
    }
}

void activations(const float* x, float* sig, float* gelu, int n) {
    if (n <= 0) return;

    float* d_x = nullptr;
    float* d_sig = nullptr;
    float* d_gelu = nullptr;

    size_t bytes = (size_t)n * sizeof(float);

    cudaMalloc(&d_x, bytes);
    cudaMalloc(&d_sig, bytes);
    cudaMalloc(&d_gelu, bytes);

    cudaMemcpy(d_x, x, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    activations_kernel<<<blocks, threads>>>(d_x, d_sig, d_gelu, n);

    cudaMemcpy(sig, d_sig, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(gelu, d_gelu, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_sig);
    cudaFree(d_gelu);
}