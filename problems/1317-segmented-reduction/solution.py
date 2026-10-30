#include <cuda_runtime.h>

#define MAXN 1024

__global__ void segsum_kernel(const float* values, const int* flags,
                              float* output, int n) {
    __shared__ float sval[MAXN];
    __shared__ int sflag[MAXN];
    __shared__ int sindex[MAXN];   // segment id per element (exclusive prefix count)
    __shared__ int sseg_pos[MAXN]; // packed output position for each segment start

    int tid = threadIdx.x;

    // Load
    if (tid < n) {
        sval[tid] = values[tid];
        sflag[tid] = flags[tid];
    }
    __syncthreads();

    // Compute segment id per element = number of flags among indices [0, tid)
    // (exclusive scan of flags) via simple parallel prefix (Hillis-Steele)
    sindex[tid] = (tid < n) ? (sflag[tid] ? 1 : 0) : 0;
    __syncthreads();

    for (int offset = 1; offset < n; offset <<= 1) {
        int v = 0;
        if (tid < n && tid >= offset) v = sindex[tid - offset];
        __syncthreads();
        if (tid < n) sindex[tid] += v;
        __syncthreads();
    }
    // Now sindex[tid] = total flags in [0, tid] inclusive.
    // Segment id (0-based) of element tid = sindex[tid] - 1.
    __syncthreads();

    // Compute packed output position for each segment start.
    // For a segment-start element t (sflag[t]==1), its position = (number of flags in [0,t]) - 1 = sindex[t]-1.
    if (tid < n) {
        if (sflag[tid]) {
            sseg_pos[sindex[tid] - 1] = tid; // record start index of segment
        }
    }
    __syncthreads();

    // Scatter-add: each element atomically adds its value to the segment's output bin.
    // The segment id for element tid is sindex[tid]-1.
    // Zero output first.
    for (int i = tid; i < n; i += blockDim.x) output[i] = 0.0f;
    __syncthreads();

    if (tid < n) {
        int seg = sindex[tid] - 1;
        atomicAdd(&output[seg], sval[tid]);
    }
}

void solve(const float* values, const int* flags, float* output, int n) {
    float* d_values = nullptr;
    int* d_flags = nullptr;
    float* d_output = nullptr;

    cudaMalloc(&d_values, n * sizeof(float));
    cudaMalloc(&d_flags, n * sizeof(int));
    cudaMalloc(&d_output, n * sizeof(float));

    cudaMemcpy(d_values, values, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_flags, flags, n * sizeof(int), cudaMemcpyHostToDevice);

    int threads = 1;
    while (threads < n) threads <<= 1;
    if (threads > 1024) threads = 1024;

    segsum_kernel<<<1, threads>>>(d_values, d_flags, d_output, n);

    cudaMemcpy(output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_values);
    cudaFree(d_flags);
    cudaFree(d_output);
}