#include <cuda_runtime.h>
#include <vector>

__global__ void index_kernel(int* out, int n) {
    // Compute this thread's global index and, if it is < n, write it to out.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = idx;
    }
}

std::vector<int> global_thread_indices(int n) {
    std::vector<int> result(n);
    if (n <= 0) return result;

    // 1. allocate device memory for n ints
    int* d_out = nullptr;
    cudaMalloc(&d_out, n * sizeof(int));

    // 2. launch the kernel with enough threads to cover n
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    index_kernel<<<blocks, threads>>>(d_out, n);

    // 3. copy the result back to the host and return it
    cudaMemcpy(result.data(), d_out, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_out);
    return result;
}