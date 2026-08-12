import torch

def engram_context_gating(h: torch.Tensor, e: torch.Tensor, W_K: torch.Tensor, W_V: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Implement Engram context-aware gating mechanism.
    
    Args:
        h: Hidden states of shape (T, d)
        e: Retrieved memory embeddings of shape (T, d_mem)
        W_K: Key projection matrix of shape (d_mem, d)
        W_V: Value projection matrix of shape (d_mem, d)
        eps: Small constant for numerical stability in RMSNorm
    
    Returns:
        Gated output of shape (T, d)
    """
    # Project memory embeddings
    # e: (T, d_mem), W_K: (d_mem, d) -> k: (T, d)
    k = e @ W_K
    # e: (T, d_mem), W_V: (d_mem, d) -> v: (T, d)
    v = e @ W_V
    
    # Apply RMSNorm to h and k
    # RMSNorm: x / sqrt(mean(x^2) + eps)
    # For h
    h_rms = torch.sqrt(torch.mean(h ** 2, dim=-1, keepdim=True) + eps)
    h_norm = h / h_rms
    
    # For k
    k_rms = torch.sqrt(torch.mean(k ** 2, dim=-1, keepdim=True) + eps)
    k_norm = k / k_rms
    
    # Compute dot product similarity
    # Sum over feature dimension: (T, d) -> (T,)
    dot_product = torch.sum(h_norm * k_norm, dim=-1)
    
    # Scale by 1/sqrt(d)
    d = h.shape[-1]
    scaled_dot = dot_product / torch.sqrt(torch.tensor(d, dtype=h.dtype, device=h.device))
    
    # Compute gate: sigmoid(scaled_dot)
    gate = torch.sigmoid(scaled_dot)
    
    # Apply gate to value projection
    # gate: (T,), v: (T, d) -> output: (T, d)
    output = gate.unsqueeze(-1) * v
    
    return output