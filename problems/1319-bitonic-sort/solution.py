#include <cuda_runtime.h>

__global__ void bitonic_sort_kernel(float* data, int n) {
    // One block, n threads, n <= 1024.
    extern __shared__ float sdata[];

    int tid = threadIdx.x;

    // Load the array into shared memory.
    sdata[tid] = data[tid];
    __syncthreads();

    // Bitonic sort network.
    for (int k = 2; k <= n; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int ixj = tid ^ j;

            if (ixj > tid) {
                // Direction: ascending when bit k of tid is clear, descending otherwise.
                // At the final stage k == n, (tid & n) == 0 for all tid, so everything ascending.
                bool ascending = ((tid & k) == 0);

                if ((sdata[tid] > sdata[ixj]) == ascending) {
                    float tmp = sdata[tid];
                    sdata[tid] = sdata[ixj];
                    sdata[ixj] = tmp;
                }
            }
            __syncthreads();
        }
    }

    // Write the sorted data back to global memory.
    data[tid] = sdata[tid];
}

void solve(const float* input, float* output, int n) {
    size_t bytes = static_cast<size_t>(n) * sizeof(float);

    float* d_data = nullptr;
    cudaMalloc(&d_data, bytes);

    cudaMemcpy(d_data, input, bytes, cudaMemcpyHostToDevice);

    // One block of n threads, n floats of shared memory.
    bitonic_sort_kernel<<<1, n, bytes>>>(d_data, n);

    cudaMemcpy(output, d_data, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_data);
}