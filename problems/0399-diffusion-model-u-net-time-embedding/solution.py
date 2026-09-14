import torch
import torch.nn.functional as F
import math

def unet_time_embedding(timesteps: list, embed_dim: int, W1: torch.Tensor, b1: torch.Tensor, W2: torch.Tensor, b2: torch.Tensor, max_period: int = 10000) -> torch.Tensor:
    """
    Compute time embeddings for a diffusion model U-Net.
    
    Args:
        timesteps: list or 1D array of shape (B,) with timestep values
        embed_dim: dimension of sinusoidal embedding (must be even)
        W1: weight matrix of first linear layer, shape (embed_dim, hidden_dim)
        b1: bias of first linear layer, shape (hidden_dim,)
        W2: weight matrix of second linear layer, shape (hidden_dim, output_dim)
        b2: bias of second linear layer, shape (output_dim,)
        max_period: controls the frequency range for sinusoidal embedding
    
    Returns:
        torch.Tensor of shape (B, output_dim) with time embeddings
    """
    # 1. Ensure timesteps is a torch.Tensor on the correct device and dtype
    if not isinstance(timesteps, torch.Tensor):
        timesteps = torch.tensor(timesteps, dtype=W1.dtype, device=W1.device)
    else:
        timesteps = timesteps.to(dtype=W1.dtype, device=W1.device)
        
    if timesteps.ndim == 0:
        timesteps = timesteps.unsqueeze(0)
        
    half_dim = embed_dim // 2
    
    # 2. Compute sinusoidal frequencies
    i = torch.arange(half_dim, dtype=W1.dtype, device=W1.device)
    freqs = torch.exp(-math.log(max_period) * i / half_dim)  # Shape: (half_dim,)
    
    # Outer product of timesteps and freqs: shape (B, half_dim)
    args = timesteps.unsqueeze(1) * freqs.unsqueeze(0)
    
    # Concatenate [sin(args), cos(args)] to form sinusoidal embedding: shape (B, embed_dim)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    
    # 3. Stage 2: Pass through two-layer MLP with SiLU activation
    h = torch.matmul(emb, W1) + b1
    h = F.silu(h)
    
    output = torch.matmul(h, W2) + b2
    
    return output