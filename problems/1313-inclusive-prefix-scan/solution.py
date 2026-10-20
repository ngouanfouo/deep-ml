#include <cuda_runtime.h>

__global__ void inclusive_scan_kernel(const float* input, float* output, int n) {
    // Two shared-memory buffers of length n: Hillis–Steele double-buffering
    // so reads and writes never alias within an iteration.
    extern __shared__ float smem[];
    float* buf0 = smem;
    float* buf1 = smem + n;

    int tid = threadIdx.x;

    // Load the input into the first buffer.
    for (int i = tid; i < n; i += blockDim.x) {
        buf0[i] = input[i];
    }
    __syncthreads();

    float* src = buf0;
    float* dst = buf1;

    // Hillis–Steele: at each step, dst[i] = src[i] + (i >= offset ? src[i-offset] : 0).
    for (int offset = 1; offset < n; offset <<= 1) {
        for (int i = tid; i < n; i += blockDim.x) {
            float val = src[i];
            if (i >= offset) {
                val += src[i - offset];
            }
            dst[i] = val;
        }
        __syncthreads();

        // Swap buffers: all writes to dst are visible (sync above); next step
        // reads from the just-written buffer and writes the other.
        float* tmp = src;
        src = dst;
        dst = tmp;
    }

    // Copy the final result (whichever buffer src points to) to global memory.
    for (int i = tid; i < n; i += blockDim.x) {
        output[i] = src[i];
    }
}

void solve(const float* input, float* output, int n) {
    size_t bytes = static_cast<size_t>(n) * sizeof(float);

    float* d_in = nullptr;
    float* d_out = nullptr;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);

    cudaMemcpy(d_in, input, bytes, cudaMemcpyHostToDevice);

    // Shared memory: two buffers of n floats each.
    size_t smem_bytes = 2 * static_cast<size_t>(n) * sizeof(float);

    int threads = n;  // n <= 1024, so a single block with n threads suffices
    inclusive_scan_kernel<<<1, threads, smem_bytes>>>(d_in, d_out, n);

    cudaMemcpy(output, d_out, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
}