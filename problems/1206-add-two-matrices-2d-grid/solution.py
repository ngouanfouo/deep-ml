#include <cuda_runtime.h>
#include <vector>

__global__ void matadd_kernel(const float* A, const float* B, float* C, int rows, int cols) {
    // Compute 2D row and column indices from thread and block indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check bounds to prevent out-of-bounds memory accesses
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        C[idx] = A[idx] + B[idx];
    }
}

std::vector<float> matrix_add(const std::vector<std::vector<float>>& A,
                              const std::vector<std::vector<float>>& B) {
    int rows = A.size();
    int cols = A[0].size();
    size_t total_elements = rows * cols;
    size_t size_bytes = total_elements * sizeof(float);

    // Flatten 2D vectors into 1D row-major arrays on the host
    std::vector<float> flat_A(total_elements);
    std::vector<float> flat_B(total_elements);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            flat_A[i * cols + j] = A[i][j];
            flat_B[i * cols + j] = B[i][j];
        }
    }

    // Allocate device memory
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    cudaMalloc((void**)&d_A, size_bytes);
    cudaMalloc((void**)&d_B, size_bytes);
    cudaMalloc((void**)&d_C, size_bytes);

    // Copy input data from host to device
    cudaMemcpy(d_A, flat_A.data(), size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, flat_B.data(), size_bytes, cudaMemcpyHostToDevice);

    // Configure 2D block and grid dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((cols + blockDim.x - 1) / blockDim.x, (rows + blockDim.y - 1) / blockDim.y);

    // Launch the CUDA kernel
    matadd_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, rows, cols);

    // Prepare host vector for result and copy back from device
    std::vector<float> C(total_elements);
    cudaMemcpy(C.data(), d_C, size_bytes, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return C;
}