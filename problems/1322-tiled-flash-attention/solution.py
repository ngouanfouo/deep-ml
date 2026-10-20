#include <cuda_runtime.h>

__global__ void flash_attn_kernel(const float* Q, const float* K, const float* V,
                                  float* O, int S, int D) {
    int i = blockIdx.x;   // query row index
    int d = threadIdx.x;  // output dimension this thread owns (0..D-1)

    // D <= 16, so q fits comfortably in shared memory.
    __shared__ float q_s[16];
    for (int k = d; k < D; k += blockDim.x) {
        q_s[k] = Q[i * D + k];
    }
    __syncthreads();

    const float scale = 1.0f / sqrtf((float)D);
    const int Bc = 4;

    float m = -1e30f;   // running max
    float l = 0.0f;     // running sum of exp(s - m)
    float o = 0.0f;     // this thread's running output accumulator (dim d)

    for (int t = 0; t < S; t += Bc) {
        int tend = t + Bc;
        if (tend > S) tend = S;
        int n = tend - t;

        // Scores for this tile (recomputed per thread; cheap and identical).
        float s[4];
        float tile_max = -1e30f;
        for (int j = 0; j < n; j++) {
            float dot = 0.0f;
            for (int k = 0; k < D; k++) {
                dot += q_s[k] * K[(t + j) * D + k];
            }
            s[j] = dot * scale;
            if (s[j] > tile_max) tile_max = s[j];
        }

        // Online-softmax merge with the tile.
        float m_new = fmaxf(m, tile_max);
        float rescale = __expf(m - m_new);
        l *= rescale;
        o *= rescale;

        for (int j = 0; j < n; j++) {
            float p = __expf(s[j] - m_new);
            l += p;
            o += p * V[(t + j) * D + d];
        }
        m = m_new;
    }

    O[i * D + d] = o / l;
}

void solve(const float* Q, const float* K, const float* V, float* O, int S, int D) {
    size_t bytes = (size_t)S * D * sizeof(float);

    float* d_Q = nullptr;
    float* d_K = nullptr;
    float* d_V = nullptr;
    float* d_O = nullptr;
    cudaMalloc(&d_Q, bytes);
    cudaMalloc(&d_K, bytes);
    cudaMalloc(&d_V, bytes);
    cudaMalloc(&d_O, bytes);

    cudaMemcpy(d_Q, Q, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V, bytes, cudaMemcpyHostToDevice);

    // One block per query row; D threads, one per output dimension.
    flash_attn_kernel<<<S, D>>>(d_Q, d_K, d_V, d_O, S, D);

    cudaMemcpy(O, d_O, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
}