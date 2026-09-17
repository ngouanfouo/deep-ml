#include <cuda_runtime.h>

#define BLOCK 256

// Same-length 1-D convolution with zero padding.
// Each block loads a contiguous chunk of `input` plus a halo of `radius`
// elements on each side into shared memory, then every thread computes one
// output element from the shared tile. Interior values are fetched from
// global memory once per block instead of once per neighbor.
__global__ void conv1d_kernel(const float* __restrict__ input,
                              const float* __restrict__ kernel,
                              float* __restrict__ output,
                              int n, int radius) {
    // Shared tile: BLOCK outputs need (BLOCK + 2*radius) inputs.
    extern __shared__ float tile[];
    // Also cache the kernel in shared memory (2*radius+1 taps).
    float* s_kernel = tile + BLOCK + 2 * radius;

    const int tile_len = BLOCK + 2 * radius;
    const int block_start = blockIdx.x * BLOCK;  // global index of tile[radius]

    // ---- Load the halo tile into shared memory ----
    // Tile position t corresponds to global index block_start - radius + t.
    for (int t = threadIdx.x; t < tile_len; t += blockDim.x) {
        int g = block_start - radius + t;
        tile[t] = (g >= 0 && g < n) ? input[g] : 0.0f;
    }

    // ---- Load the kernel taps into shared memory ----
    for (int t = threadIdx.x; t < 2 * radius + 1; t += blockDim.x) {
        s_kernel[t] = kernel[t];
    }

    __syncthreads();

    // ---- Each thread computes one output element ----
    int i = block_start + threadIdx.x;
    if (i < n) {
        float acc = 0.0f;
        // tile index for x[i + k] is (threadIdx.x + radius + k).
        for (int k = -radius; k <= radius; ++k) {
            acc += tile[threadIdx.x + radius + k] * s_kernel[k + radius];
        }
        output[i] = acc;
    }
}

void solve(const float* input, const float* kernel, float* output,
           int n, int radius) {
    if (n <= 0 || radius < 0) return;

    const size_t bytes_in = static_cast<size_t>(n) * sizeof(float);
    const size_t bytes_out = static_cast<size_t>(n) * sizeof(float);
    const size_t bytes_k = static_cast<size_t>(2 * radius + 1) * sizeof(float);

    // Allocate device memory
    float* d_in = nullptr;
    float* d_k = nullptr;
    float* d_out = nullptr;
    cudaMalloc(&d_in, bytes_in);
    cudaMalloc(&d_k, bytes_k);
    cudaMalloc(&d_out, bytes_out);

    // Copy inputs to device
    cudaMemcpy(d_in, input, bytes_in, cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, kernel, bytes_k, cudaMemcpyHostToDevice);

    // Launch: one block per chunk of BLOCK outputs.
    const int blocks = (n + BLOCK - 1) / BLOCK;

    // Shared memory = tile (BLOCK + 2*radius) + kernel (2*radius + 1) floats.
    const size_t smem_bytes =
        (BLOCK + 2 * radius + 2 * radius + 1) * sizeof(float);

    conv1d_kernel<<<blocks, BLOCK, smem_bytes>>>(d_in, d_k, d_out, n, radius);

    // Copy result back to host
    cudaMemcpy(output, d_out, bytes_out, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_in);
    cudaFree(d_k);
    cudaFree(d_out);
}