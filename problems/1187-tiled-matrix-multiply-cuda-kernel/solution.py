#include <cuda_runtime.h>
#include <algorithm>

#define BLOCK_SIZE 16

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // Shared memory for tiles of A and B
    __shared__ float s_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_B[BLOCK_SIZE][BLOCK_SIZE];

    // Block and thread indices
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Row and column of C that this thread is responsible for
    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;

    float sum = 0.0f;

    // Loop over all tiles required to compute the C element
    int numTiles = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int t = 0; t < numTiles; ++t) {
        // Load tile of A into shared memory (with out-of-bounds checking)
        int a_row = row;
        int a_col = t * BLOCK_SIZE + tx;
        if (a_row < M && a_col < K) {
            s_A[ty][tx] = A[a_row * K + a_col];
        } else {
            s_A[ty][tx] = 0.0f;
        }

        // Load tile of B into shared memory (with out-of-bounds checking)
        int b_row = t * BLOCK_SIZE + ty;
        int b_col = col;
        if (b_row < K && b_col < N) {
            s_B[ty][tx] = B[b_row * N + b_col];
        } else {
            s_B[ty][tx] = 0.0f;
        }

        // Wait for all threads in the block to load their tile elements
        __syncthreads();

        // Compute partial dot product for this tile
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += s_A[ty][k] * s_B[k][tx];
        }

        // Wait for threads to finish using current tiles before loading the next ones
        __syncthreads();
    }

    // Write the final result back to global memory
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    size_t size_A = (size_t)M * K * sizeof(float);
    size_t size_B = (size_t)K * N * sizeof(float);
    size_t size_C = (size_t)M * N * sizeof(float);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

    // 1. Allocate device memory
    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_B);
    cudaMalloc((void**)&d_C, size_C);

    // 2. Copy inputs from host to device
    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);

    // 3. Define grid and block dimensions
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // 4. Launch the CUDA kernel
    matmul_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);

    // 5. Copy the result back to host memory
    cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);

    // 6. Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}