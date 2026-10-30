#include <cuda_runtime.h>

__global__ void rmsnorm_kernel(const float* input, const float* gamma,
                               float* output, int rows, int cols, float eps) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* in_row = input + (long)row * cols;
    float* out_row = output + (long)row * cols;

    int tid = threadIdx.x;
    int stride = blockDim.x;

    // Sum of squares
    float ss = 0.0f;
    for (int j = tid; j < cols; j += stride) {
        float v = in_row[j];
        ss += v * v;
    }

    // Block reduction in shared memory
    __shared__ float sdata[1024];
    sdata[tid] = ss;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    __shared__ float inv_rms;
    if (tid == 0) {
        inv_rms = rsqrtf(sdata[0] / (float)cols + eps);
    }
    __syncthreads();

    float scale = inv_rms;
    for (int j = tid; j < cols; j += stride) {
        out_row[j] = in_row[j] * scale * gamma[j];
    }
}

void solve(const float* input, const float* gamma, float* output,
           int rows, int cols, float eps) {
    float* d_input = nullptr;
    float* d_gamma = nullptr;
    float* d_output = nullptr;

    size_t mat_bytes = (size_t)rows * cols * sizeof(float);

    cudaMalloc(&d_input, mat_bytes);
    cudaMalloc(&d_gamma, cols * sizeof(float));
    cudaMalloc(&d_output, mat_bytes);

    cudaMemcpy(d_input, input, mat_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, gamma, cols * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 1;
    while (threads < cols) threads <<= 1;
    if (threads < 32) threads = 32;
    if (threads > 1024) threads = 1024;

    rmsnorm_kernel<<<rows, threads>>>(d_input, d_gamma, d_output, rows, cols, eps);

    cudaMemcpy(output, d_output, mat_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_gamma);
    cudaFree(d_output);
}