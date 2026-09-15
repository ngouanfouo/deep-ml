#include <cuda_runtime.h>

#define BLOCK_SIZE 256

__global__ void kernel(const float* in, float* out, int N) {
    __shared__ float sdata[BLOCK_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load elements into shared memory with bounds checking
    float sum = 0.0f;
    if (idx < N) {
        sum = in[idx];
    }
    sdata[tid] = sum;
    __syncthreads();

    // Do reduction in shared memory (halving active threads each step)
    for (unsigned int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write the result of this block to the global output using atomicAdd
    if (tid == 0) {
        atomicAdd(out, sdata[0]);
    }
}

void solve(const float* input, float* output, int N) {
    size_t size_input = (size_t)N * sizeof(float);
    float *d_in = nullptr;
    float *d_out = nullptr;

    // 1. Allocate device memory
    cudaMalloc((void**)&d_in, size_input);
    cudaMalloc((void**)&d_out, sizeof(float));

    // 2. Copy input data from host to device
    cudaMemcpy(d_in, input, size_input, cudaMemcpyHostToDevice);

    // Initialize output accumulator on device to 0.0f
    float zero = 0.0f;
    cudaMemcpy(d_out, &zero, sizeof(float), cudaMemcpyHostToDevice);

    // 3. Configure grid and block dimensions
    int block_size = BLOCK_SIZE;
    int grid_size = (N + block_size - 1) / block_size;

    // 4. Launch the CUDA kernel
    kernel<<<grid_size, block_size>>>(d_in, d_out, N);

    // 5. Copy the final result back to host memory
    cudaMemcpy(output, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    // 6. Free device memory
    cudaFree(d_in);
    cudaFree(d_out);
}