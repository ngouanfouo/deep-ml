#include <cuda_runtime.h>
#include <vector>

#define BLOCK_SIZE 256

__global__ void dot_kernel(const float* a, const float* b, float* out, int n) {
    // Shared memory for the tree reduction.
    __shared__ float cache[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 1. Each thread computes its elementwise product into shared memory.
    //    Out-of-range threads contribute 0 so the reduction stays correct.
    cache[tid] = (idx < n) ? a[idx] * b[idx] : 0.0f;
    __syncthreads();

    // 2. Tree reduction in shared memory: halve the active threads each step.
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            cache[tid] += cache[tid + s];
        }
        __syncthreads();
    }

    // 3. Thread 0 writes the block's partial sum.
    if (tid == 0) {
        out[blockIdx.x] = cache[0];
    }
}

float dot_product(const std::vector<float>& a, const std::vector<float>& b) {
    const int n = static_cast<int>(a.size());
    if (n == 0) return 0.0f;

    const size_t bytes = n * sizeof(float);

    // 1. Allocate device memory
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_out = nullptr;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);

    // Number of blocks needed (assumes n <= BLOCK_SIZE, so blocks == 1 here).
    const int threads = BLOCK_SIZE;
    const int blocks = (n + threads - 1) / threads;
    cudaMalloc(&d_out, blocks * sizeof(float));

    // 2. Copy inputs to the device
    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    // 3. Launch the kernel
    dot_kernel<<<blocks, threads>>>(d_a, d_b, d_out, n);

    // 4. Copy back the per-block partial sums and finish the sum on the host.
    //    (With the stated n <= 256 constraint, blocks == 1 and this is already
    //     the full dot product.)
    std::vector<float> partials(blocks);
    cudaMemcpy(partials.data(), d_out, blocks * sizeof(float), cudaMemcpyDeviceToHost);

    float result = 0.0f;
    for (float p : partials) result += p;

    // 5. Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    return result;
}