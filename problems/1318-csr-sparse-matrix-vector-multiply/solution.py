#include <cuda_runtime.h>

__global__ void csr_matvec_kernel(const int* row_ptr, const int* col_idx,
                                  const float* vals, const float* x,
                                  float* y, int M, int nnz, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) return;

    int start = row_ptr[i];
    int end = row_ptr[i + 1];

    float acc = 0.0f;
    for (int k = start; k < end; ++k) {
        acc += vals[k] * x[col_idx[k]];
    }
    y[i] = acc;
}

void solve(const int* row_ptr, const int* col_idx, const float* vals,
           const float* x, float* y, int M, int nnz, int N) {
    int* d_row_ptr = nullptr;
    int* d_col_idx = nullptr;
    float* d_vals = nullptr;
    float* d_x = nullptr;
    float* d_y = nullptr;

    cudaMalloc(&d_row_ptr, (M + 1) * sizeof(int));
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, M * sizeof(float));

    cudaMemcpy(d_row_ptr, row_ptr, (M + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);

    if (nnz > 0) {
        cudaMalloc(&d_col_idx, nnz * sizeof(int));
        cudaMalloc(&d_vals, nnz * sizeof(float));
        cudaMemcpy(d_col_idx, col_idx, nnz * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_vals, vals, nnz * sizeof(float), cudaMemcpyHostToDevice);
    }

    int threads = 128;
    int blocks = (M + threads - 1) / threads;
    csr_matvec_kernel<<<blocks, threads>>>(d_row_ptr, d_col_idx, d_vals, d_x, d_y, M, nnz, N);

    cudaMemcpy(y, d_y, M * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_row_ptr);
    cudaFree(d_x);
    cudaFree(d_y);
    if (nnz > 0) {
        cudaFree(d_col_idx);
        cudaFree(d_vals);
    }
}