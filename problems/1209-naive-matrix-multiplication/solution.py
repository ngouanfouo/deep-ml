#include <cuda_runtime.h>
#include <vector>

__global__ void matmul_kernel(const float* A, const float* B, float* C,
                              int M, int K, int N) {
    // Each thread computes one output element C[row][col].
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            acc += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = acc;
    }
}

std::vector<float> matmul(const std::vector<std::vector<float>>& A,
                          const std::vector<std::vector<float>>& B) {
    const int M = static_cast<int>(A.size());
    const int K = (M > 0) ? static_cast<int>(A[0].size()) : 0;
    const int N = (K > 0) ? static_cast<int>(B[0].size()) : 0;

    std::vector<float> C(M * N, 0.0f);
    if (M == 0 || K == 0 || N == 0) return C;

    // Flatten A and B into contiguous row-major host buffers.
    std::vector<float> h_A(M * K);
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < K; ++j)
            h_A[i * K + j] = A[i][j];

    std::vector<float> h_B(K * N);
    for (int i = 0; i < K; ++i)
        for (int j = 0; j < N; ++j)
            h_B[i * N + j] = B[i][j];

    // 1. Allocate device memory
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    // 2. Copy inputs to the device
    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);

    // 3. Launch the kernel: one thread per output element.
    //    Use a 2D block/grid so each thread maps naturally to (row, col).
    const int TILE = 16;
    dim3 threads(TILE, TILE);
    dim3 blocks((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    matmul_kernel<<<blocks, threads>>>(d_A, d_B, d_C, M, K, N);

    // 4. Copy the result back to the host
    cudaMemcpy(C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // 5. Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return C;
}