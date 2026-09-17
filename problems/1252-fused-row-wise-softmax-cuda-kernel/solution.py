#include <cuda_runtime.h>
#include <cfloat>
#include <math.h>

// One thread block per row. Block size is a power of two >= cols.
__global__ void fused_softmax_kernel(const float* input, float* output,
                                     int rows, int cols) {
    // Shared memory layout:
    //   s_data[0 .. blockDim.x-1] : row values, then exp(x - max)
    //   s_reduce[0 .. blockDim.x-1] : scratch for tree reductions
    extern __shared__ float smem[];
    float* s_data   = smem;                 // blockDim.x floats
    float* s_reduce = smem + blockDim.x;    // blockDim.x floats

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int base = row * cols;

    // ---- 1) Load row values (identity = -FLT_MAX for max) ----
    float v = -FLT_MAX;
    if (tid < cols) {
        v = input[base + tid];
    }
    s_data[tid] = v;
    s_reduce[tid] = v;
    __syncthreads();

    // ---- 2) Shared-memory max reduction ----
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            float other = s_reduce[tid + s];
            if (other > s_reduce[tid]) s_reduce[tid] = other;
        }
        __syncthreads();
    }
    float row_max = s_reduce[0];
    __syncthreads(); // make sure everyone has read s_reduce[0] before reuse

    // ---- 3) exp(x - max); identity = 0 for sum ----
    float e = 0.0f;
    if (tid < cols) {
        e = expf(s_data[tid] - row_max);
    }
    s_data[tid] = e;
    s_reduce[tid] = e;
    __syncthreads();

    // ---- 4) Shared-memory sum reduction ----
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            s_reduce[tid] += s_reduce[tid + s];
        }
        __syncthreads();
    }
    float row_sum = s_reduce[0];
    float inv_sum = 1.0f / row_sum;

    // ---- 5) Normalize and write out ----
    if (tid < cols) {
        output[base + tid] = s_data[tid] * inv_sum;
    }
}

void solve(const float* input, float* output, int rows, int cols) {
    if (rows <= 0 || cols <= 0) return;

    const size_t bytes = static_cast<size_t>(rows) * cols * sizeof(float);

    // Allocate device memory
    float* d_in = nullptr;
    float* d_out = nullptr;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);

    // Copy input to device
    cudaMemcpy(d_in, input, bytes, cudaMemcpyHostToDevice);

    // Choose a power-of-two block size >= cols (and <= 1024).
    int threads = 1;
    while (threads < cols) threads <<= 1;

    // One block per row.
    dim3 grid(rows);
    dim3 block(threads);

    // Shared memory: two arrays of `threads` floats.
    const size_t smem_bytes = 2 * threads * sizeof(float);

    fused_softmax_kernel<<<grid, block, smem_bytes>>>(d_in, d_out, rows, cols);

    // Copy result back to host
    cudaMemcpy(output, d_out, bytes, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_in);
    cudaFree(d_out);
}