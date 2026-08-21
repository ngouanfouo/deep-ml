import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn,
                  stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                  BLOCK_K: tl.constexpr):
    # Get program ID for this tile
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Calculate starting positions for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Create masks for boundary conditions
    m_mask = offs_m < M
    n_mask = offs_n < N
    
    # Initialize accumulator to float32 for precision
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Iterate over K dimension in chunks of BLOCK_K
    for k in range(0, K, BLOCK_K):
        # Load a tile of A (BLOCK_M, BLOCK_K)
        offs_k = k + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K
        
        # Load A tile with masks
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        a_tile = tl.load(a_ptrs, mask=(m_mask[:, None] & k_mask[None, :]), other=0.0)
        
        # Load B tile (BLOCK_K, BLOCK_N)
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        b_tile = tl.load(b_ptrs, mask=(k_mask[:, None] & n_mask[None, :]), other=0.0)
        
        # Accumulate the result
        acc += tl.dot(a_tile, b_tile)
    
    # Cast accumulator back to input dtype
    output_dtype = a_ptr.dtype.element_ty
    c_tile = acc.to(output_dtype)
    
    # Store the result tile
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c_tile, mask=(m_mask[:, None] & n_mask[None, :]))


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Matrix multiplication using Triton with tiled GEMM."""
    # Get dimensions
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"Matrix dimensions mismatch: a shape {a.shape}, b shape {b.shape}"
    
    # Allocate output tensor
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # Tile sizes
    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 32
    
    # Grid over output tiles
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    # Launch kernel
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K
    )
    
    return c