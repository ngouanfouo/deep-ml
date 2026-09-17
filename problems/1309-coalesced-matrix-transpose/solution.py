#include <cuda_runtime.h>

#define TILE 16

// Tiled transpose: A is M x N row-major, B is N x M row-major.
// B[j*M + i] = A[i*N + j].
// Each block handles a TILE x TILE tile of A, restaging it through
// shared memory so both the global load and the global store are coalesced.
__global__ void transpose_kernel(const float* A, float* B, int M, int N) {
    __shared__ float tile[TILE][TILE + 1];  // +1 padding to avoid bank conflicts

    // Block covers A rows [blockIdx.y*TILE .. +TILE) and cols [blockIdx.x*TILE .. +TILE).
    int i = blockIdx.y * TILE + threadIdx.y;  // A row
    int j = blockIdx.x * TILE + threadIdx.x;  // A col

    // Coalesced load from A: consecutive threadIdx.x -> consecutive j -> consecutive A[i*N+j].
    if (i < M && j < N) {
        tile[threadIdx.y][threadIdx.x] = A[i * N + j];
    }

    __syncthreads();

    // Transposed coordinates in B.
    int bi = blockIdx.x * TILE + threadIdx.y;  // B row = A col
    int bj = blockIdx.y * TILE + threadIdx.x;  // B col = A row

    // Coalesced store to B: consecutive threadIdx.x -> consecutive bj -> consecutive B[bi*M+bj].
    // Read from shared memory with swapped indices (the actual transpose).
    if (bi < N && bj < M) {
        B[bi * M + bj] = tile[threadIdx.x][threadIdx.y];
    }
}

void solve(const float* A, float* B, int M, int N) {
    if (M <= 0 || N <= 0) return;

    const size_t bytesA = static_cast<size_t>(M) * N * sizeof(float);
    const size_t bytesB = static_cast<size_t>(N) * M * sizeof(float);

    // Allocate device memory
    float* d_A = nullptr;
    float* d_B = nullptr;
    cudaMalloc(&d_A, bytesA);
    cudaMalloc(&d_B, bytesB);

    // Copy input to device
    cudaMemcpy(d_A, A, bytesA, cudaMemcpyHostToDevice);

    // Launch: one thread per tile element, grid covers all tiles of A.
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    transpose_kernel<<<grid, block>>>(d_A, d_B, M, N);

    // Copy result back to host
    cudaMemcpy(B, d_B, bytesB, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
}