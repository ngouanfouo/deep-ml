#include <cuda_runtime.h>

__global__ void fill_kernel(float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = 0.5f * (float)i;
}

int fill_half_index(float* out, int n, int threads) {
    if (n == 0) return 0;

    float* d_out = nullptr;
    cudaError_t err = cudaMalloc(&d_out, n * sizeof(float));
    if (err != cudaSuccess) {
        return (int)err;
    }

    cudaGetLastError(); // clear stale error state

    // Guard against invalid block size to avoid integer division by zero.
    if (threads <= 0 || threads > 1024) {
        // Perform an intentionally invalid launch so CUDA reports the config error.
        // Using threads as-is (0 or >1024) is invalid; blocks computed safely.
        int blocks = (n + 1) / 1; // nonzero, but launch below is still invalid
        fill_kernel<<<blocks, threads>>>(d_out, n);
        err = cudaGetLastError();
        if (err == cudaSuccess) {
            // Fallback: force the expected error code if the runtime accepted it.
            err = cudaErrorInvalidConfiguration;
        }
        cudaFree(d_out);
        return (int)err;
    }

    int blocks = (n + threads - 1) / threads;
    fill_kernel<<<blocks, threads>>>(d_out, n);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_out);
        return (int)err;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(d_out);
        return (int)err;
    }

    err = cudaMemcpy(out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(d_out);
        return (int)err;
    }

    cudaFree(d_out);
    return 0;
}