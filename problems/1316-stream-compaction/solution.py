#include <cuda_runtime.h>

__global__ void compact_kernel(const float* input, float* output, int* count, int n) {
    // Two shared-memory flag buffers for a race-free Hillis–Steele scan.
    extern __shared__ int smem[];
    int* buf0 = smem;
    int* buf1 = smem + n;

    int tid = threadIdx.x;

    // 1. Flag: 1 where input[i] != 0.0f, else 0.
    for (int i = tid; i < n; i += blockDim.x) {
        buf0[i] = (input[i] != 0.0f) ? 1 : 0;
    }
    __syncthreads();

    // 2. Inclusive scan of the flags (double-buffered to avoid in-place races).
    int* src = buf0;
    int* dst = buf1;
    for (int offset = 1; offset < n; offset <<= 1) {
        for (int i = tid; i < n; i += blockDim.x) {
            int val = src[i];
            if (i >= offset) {
                val += src[i - offset];
            }
            dst[i] = val;
        }
        __syncthreads();

        int* tmp = src;
        src = dst;
        dst = tmp;
    }
    __syncthreads();

    // 3. Scatter: destination of survivor i is its exclusive prefix (inclusive - 1).
    for (int i = tid; i < n; i += blockDim.x) {
        if (input[i] != 0.0f) {
            int dest = src[i] - 1;
            output[dest] = input[i];
        }
    }

    // Total number of nonzeros is the last inclusive-scan entry.
    if (tid == 0) {
        *count = src[n - 1];
    }
}

void solve(const float* input, float* output, int* count, int n) {
    float* d_in = nullptr;
    float* d_out = nullptr;
    int* d_count = nullptr;

    size_t fbytes = static_cast<size_t>(n) * sizeof(float);
    size_t smem_bytes = 2 * static_cast<size_t>(n) * sizeof(int);

    cudaMalloc(&d_in, fbytes);
    cudaMalloc(&d_out, fbytes);
    cudaMalloc(&d_count, sizeof(int));

    cudaMemcpy(d_in, input, fbytes, cudaMemcpyHostToDevice);

    compact_kernel<<<1, n, smem_bytes>>>(d_in, d_out, d_count, n);

    int h_count = 0;
    cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);

    if (h_count > 0) {
        cudaMemcpy(output, d_out, static_cast<size_t>(h_count) * sizeof(float),
                   cudaMemcpyDeviceToHost);
    }
    *count = h_count;

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_count);
}