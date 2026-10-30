#include <cuda_runtime.h>

#define MAX_S 16
#define MAX_D 16

__global__ void attention_kernel(const float* Q, const float* K, const float* V,
                                 float* O, int S, int D) {
    int qi = blockIdx.x;
    if (qi >= S) return;

    __shared__ float scores[MAX_S];
    __shared__ float q_row[MAX_D];

    int tid = threadIdx.x;

    // Load query row into shared memory
    if (tid < D) q_row[tid] = Q[qi * D + tid];
    __syncthreads();

    float scale = rsqrtf((float)D);

    // Compute scores[qi][j] = (Q[qi] . K[j]) * scale for each key j
    for (int j = tid; j < S; j += blockDim.x) {
        float dot = 0.0f;
        for (int d = 0; d < D; ++d) {
            dot += q_row[d] * K[j * D + d];
        }
        scores[j] = dot * scale;
    }
    __syncthreads();

    // Row max (parallel reduction in shared memory)
    __shared__ float s_red[MAX_S];
    if (tid < S) s_red[tid] = scores[tid];
    __syncthreads();
    for (int s = S >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_red[tid + s] > s_red[tid]) s_red[tid] = s_red[tid + s];
        }
        __syncthreads();
    }
    __shared__ float row_max;
    if (tid == 0) row_max = s_red[0];
    __syncthreads();

    // exp and sum
    float local_sum = 0.0f;
    for (int j = tid; j < S; j += blockDim.x) {
        float e = expf(scores[j] - row_max);
        scores[j] = e;
        local_sum += e;
    }
    // Sum reduction
    if (tid < S) s_red[tid] = (tid < S) ? 0.0f : 0.0f;
    __syncthreads();
    // Use shared accumulator per thread via atomic in shared
    __shared__ float s_sum;
    if (tid == 0) s_sum = 0.0f;
    __syncthreads();
    // Reduce local sums across threads: use shared array indexed by tid (blockDim <= 1024)
    // Simpler: each thread that processed j adds to a shared atomic
    for (int j = tid; j < S; j += blockDim.x) {
        atomicAdd(&s_sum, scores[j]);
    }
    __syncthreads();

    // Normalize scores
    float inv_sum = 1.0f / s_sum;
    for (int j = tid; j < S; j += blockDim.x) {
        scores[j] *= inv_sum;
    }
    __syncthreads();

    // Compute output: O[qi][d] = sum_j scores[j] * V[j][d]
    for (int d = tid; d < D; d += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j < S; ++j) {
            acc += scores[j] * V[j * D + d];
        }
        O[qi * D + d] = acc;
    }
}

void solve(const float* Q, const float* K, const float* V, float* O, int S, int D) {
    float* d_Q = nullptr;
    float* d_K = nullptr;
    float* d_V = nullptr;
    float* d_O = nullptr;

    size_t bytes = (size_t)S * D * sizeof(float);

    cudaMalloc(&d_Q, bytes);
    cudaMalloc(&d_K, bytes);
    cudaMalloc(&d_V, bytes);
    cudaMalloc(&d_O, bytes);

    cudaMemcpy(d_Q, Q, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V, bytes, cudaMemcpyHostToDevice);

    int threads = 32;
    attention_kernel<<<S, threads>>>(d_Q, d_K, d_V, d_O, S, D);

    cudaMemcpy(O, d_O, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
}