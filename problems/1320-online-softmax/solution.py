#include <cuda_runtime.h>

__global__ void online_softmax_kernel(const float* input, float* output, int rows, int cols) {
    int row = blockIdx.x;
    const float* x = input + (size_t)row * cols;
    float* y = output + (size_t)row * cols;

    const int TILE = blockDim.x;          // 32
    extern __shared__ float sdata[];      // TILE floats

    __shared__ float s_max;               // running max m
    __shared__ float s_sum;               // running l
    __shared__ float s_mt;                // tile max m_t
    __shared__ float s_st;                // tile sum sum exp(x - m_t)

    int tid = threadIdx.x;

    if (tid == 0) {
        s_max = -1e30f;                   // acts as -inf
        s_sum = 0.0f;
    }
    __syncthreads();

    // ---- Pass 1: online recurrence over tiles of TILE elements ----
    for (int start = 0; start < cols; start += TILE) {
        int idx = start + tid;
        float val = (idx < cols) ? x[idx] : -1e30f;

        // --- local max m_t ---
        sdata[tid] = val;
        __syncthreads();
        for (int stride = TILE >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
            }
            __syncthreads();
        }
        if (tid == 0) s_mt = sdata[0];
        __syncthreads();

        float mt = s_mt;

        // --- local sum sum exp(x - m_t) ---
        float e = (idx < cols) ? __expf(val - mt) : 0.0f;
        sdata[tid] = e;
        __syncthreads();
        for (int stride = TILE >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                sdata[tid] += sdata[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0) s_st = sdata[0];
        __syncthreads();

        // --- merge tile into running (m, l) ---
        if (tid == 0) {
            float m_new = fmaxf(s_max, mt);
            s_sum = s_sum * __expf(s_max - m_new) + s_st * __expf(mt - m_new);
            s_max = m_new;
        }
        __syncthreads();
    }

    // ---- Pass 2: write exp(x - m) / l ----
    for (int idx = tid; idx < cols; idx += blockDim.x) {
        y[idx] = __expf(x[idx] - s_max) / s_sum;
    }
}

void solve(const float* input, float* output, int rows, int cols) {
    size_t bytes = (size_t)rows * cols * sizeof(float);

    float* d_in = nullptr;
    float* d_out = nullptr;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);

    cudaMemcpy(d_in, input, bytes, cudaMemcpyHostToDevice);

    const int TILE = 32;
    size_t smem_bytes = TILE * sizeof(float);

    online_softmax_kernel<<<rows, TILE, smem_bytes>>>(d_in, d_out, rows, cols);

    cudaMemcpy(output, d_out, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
}