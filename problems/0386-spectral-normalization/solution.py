import torch
from typing import Optional, Tuple

def spectral_normalization(
    W: torch.Tensor,
    num_iters: int = 10,
    u_init: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, float]:
    """
    Apply spectral normalization to a weight matrix using power iteration.

    Args:
        W: Weight matrix of shape (m, n) as a torch.Tensor
        num_iters: Number of power iteration steps
        u_init: Optional initial vector of shape (m,) as a torch.Tensor

    Returns:
        Tuple of (W_sn, sigma):
            W_sn: Spectrally normalized weight matrix as a torch.Tensor
            sigma: Estimated spectral norm (largest singular value) as a float
    """
    m, n = W.shape
    
    # 1. Initialize left singular vector u, ensuring it matches W's dtype and device
    if u_init is not None:
        u = u_init.clone().to(dtype=W.dtype, device=W.device)
    else:
        torch.manual_seed(0)
        u = torch.randn(m, dtype=W.dtype, device=W.device)
        
    # Normalize u to unit length
    u = u / (u.norm() + 1e-12)
    
    v = None
    # 2. Perform power iterations
    for _ in range(num_iters):
        # v = W^T u
        v = torch.matmul(u, W)
        v = v / (v.norm() + 1e-12)
        
        # u = W v
        u = torch.matmul(W, v)
        u = u / (u.norm() + 1e-12)
        
    # 3. Compute the spectral norm sigma = u^T W v
    sigma = torch.dot(u, torch.matmul(W, v)).item()
    
    # 4. Compute the spectrally normalized weight matrix
    W_sn = W / sigma
    
    return W_sn, float(sigma)