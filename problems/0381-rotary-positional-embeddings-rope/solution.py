import torch

def apply_rope(x: torch.Tensor, positions: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """
    Apply Rotary Positional Embeddings (RoPE) to input embeddings.
    
    Args:
        x: Input embeddings of shape (seq_len, d), d must be even
        positions: Position indices of shape (seq_len,)
        base: Base for frequency computation (default: 10000.0)
    
    Returns:
        Embeddings with rotary positional encoding applied, shape (seq_len, d)
    """
    seq_len, d = x.shape
    if d % 2 != 0:
        raise ValueError("Dimension d must be even for RoPE.")
    
    # 1. Compute frequencies for each dimension pair: theta_i = 1 / (base ** (2i / d))
    i = torch.arange(0, d, 2, dtype=torch.float32, device=x.device)
    freqs = 1.0 / (base ** (i / d))
    
    # 2. Compute angles for all positions: shape (seq_len, d/2)
    angles = torch.outer(positions.float(), freqs)
    
    # 3. Compute cosine and sine values for the rotation
    cos_vals = torch.cos(angles)
    sin_vals = torch.sin(angles)
    
    # 4. Reshape x into pairs of dimensions: shape (seq_len, d/2, 2)
    x_pairs = x.view(seq_len, d // 2, 2)
    x_even = x_pairs[..., 0]  # Even indices (e.g., 0, 2, ...)
    x_odd = x_pairs[..., 1]   # Odd indices (e.g., 1, 3, ...)
    
    # 5. Apply the 2D rotation equations
    out_even = x_even * cos_vals - x_odd * sin_vals
    out_odd = x_even * sin_vals + x_odd * cos_vals
    
    # 6. Interleave the rotated pairs back into the original shape (seq_len, d)
    output = torch.stack([out_even, out_odd], dim=-1).view(seq_len, d)
    
    return output