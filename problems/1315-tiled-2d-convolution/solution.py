#include <cuda_runtime.h>

#define TILE 16
#define MAX_R 2
#define MAX_K (2 * MAX_R + 1)

__global__ void conv2d_kernel(const float* input, const float* kernel,
                              float* output, int H, int W, int R) {
    int K = 2 * R + 1;

    // Shared tile with halo
    __shared__ float tile[TILE + 2 * MAX_R][TILE + 2 * MAX_R];
    __shared__ float sh_kernel[MAX_K * MAX_K];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int tile_w = TILE + 2 * R;

    // Base output coords for this block
    int base_i = blockIdx.y * TILE;
    int base_j = blockIdx.x * TILE;

    // Load kernel into shared memory
    int ksize = K * K;
    for (int idx = ty * TILE + tx; idx < ksize; idx += TILE * TILE) {
        sh_kernel[idx] = kernel[idx];
    }

    // Load input tile + halo into shared memory
    for (int ly = ty; ly < tile_w; ly += TILE) {
        for (int lx = tx; lx < tile_w; lx += TILE) {
            int gi = base_i + ly - R;
            int gj = base_j + lx - R;
            float v = 0.0f;
            if (gi >= 0 && gi < H && gj >= 0 && gj < W) {
                v = input[gi * W + gj];
            }
            tile[ly][lx] = v;
        }
    }

    __syncthreads();

    // Compute output
    int oi = base_i + ty;
    int oj = base_j + tx;
    if (oi < H && oj < W) {
        float acc = 0.0f;
        for (int u = 0; u < K; ++u) {
            for (int v = 0; v < K; ++v) {
                acc += tile[ty + u][tx + v] * sh_kernel[u * K + v];
            }
        }
        output[oi * W + oj] = acc;
    }
}

void solve(const float* input, const float* kernel, float* output,
           int H, int W, int radius) {
    int ksize = (2 * radius + 1) * (2 * radius + 1);

    float* d_input = nullptr;
    float* d_kernel = nullptr;
    float* d_output = nullptr;

    size_t img_bytes = (size_t)H * W * sizeof(float);

    cudaMalloc(&d_input, img_bytes);
    cudaMalloc(&d_kernel, ksize * sizeof(float));
    cudaMalloc(&d_output, img_bytes);

    cudaMemcpy(d_input, input, img_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, ksize * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(TILE, TILE);
    dim3 grid((W + TILE - 1) / TILE, (H + TILE - 1) / TILE);

    conv2d_kernel<<<grid, block>>>(d_input, d_kernel, d_output, H, W, radius);

    cudaMemcpy(output, d_output, img_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
}