#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>

__global__ void kernel(const float* input, const float* bias, float* output, int rows, int cols) {
    // Calculate global flat index for the element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = rows * cols;

    if (idx < total_elements) {
        // Determine the column index for the current element
        int col = idx % cols;

        // Add corresponding column bias and apply ReLU activation: max(0, x)
        float val = input[idx] + bias[col];
        output[idx] = fmaxf(0.0f, val);
    }
}

void solve(const float* input, const float* bias, float* output, int rows, int cols) {
    size_t size_matrix = (size_t)rows * cols * sizeof(float);
    size_t size_bias = (size_t)cols * sizeof(float);

    float *d_input = nullptr, *d_bias = nullptr, *d_output = nullptr;

    // 1. Allocate device memory
    cudaMalloc((void**)&d_input, size_matrix);
    cudaMalloc((void**)&d_bias, size_bias);
    cudaMalloc((void**)&d_output, size_matrix);

    // 2. Copy inputs from host to device
    cudaMemcpy(d_input, input, size_matrix, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, size_bias, cudaMemcpyHostToDevice);

    // 3. Configure grid and block dimensions
    int total_elements = rows * cols;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;

    // 4. Launch the CUDA kernel
    kernel<<<grid_size, block_size>>>(d_input, d_bias, d_output, rows, cols);

    // 5. Copy the result back to host memory
    cudaMemcpy(output, d_output, size_matrix, cudaMemcpyDeviceToHost);

    // 6. Free device memory
    cudaFree(d_input);
    cudaFree(d_bias);
    cudaFree(d_output);
}