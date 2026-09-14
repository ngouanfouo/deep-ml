import torch
import math

def kv_cache_attention_step(x_new: torch.Tensor, W_Q: torch.Tensor, W_K: torch.Tensor, W_V: torch.Tensor, cache: tuple) -> tuple:
    """
    Perform a single attention step with KV caching.
    
    Args:
        x_new: New token embedding, shape (d_model,)
        W_Q: Query projection matrix, shape (d_model, d_k)
        W_K: Key projection matrix, shape (d_model, d_k)
        W_V: Value projection matrix, shape (d_model, d_v)
        cache: Tuple (K_cache, V_cache) of tensors or None if first step
    
    Returns:
        Tuple (output, updated_cache) where output is shape (d_v,)
        and updated_cache is (K_new, V_new)
    """
    # 1. Project new token into query, key, and value vectors
    q = torch.matmul(x_new, W_Q)  # Shape: (d_k,)
    k = torch.matmul(x_new, W_K)  # Shape: (d_k,)
    v = torch.matmul(x_new, W_V)  # Shape: (d_v,)
    
    # Add batch/sequence dimension for concatenation: shape (1, d_k) or (1, d_v)
    k_expanded = k.unsqueeze(0)
    v_expanded = v.unsqueeze(0)
    
    # 2. Append new key and value to the existing cache
    if cache is None:
        K_new = k_expanded
        V_new = v_expanded
    else:
        K_cache, V_cache = cache
        K_new = torch.cat([K_cache, k_expanded], dim=0)
        V_new = torch.cat([V_cache, v_expanded], dim=0)
        
    # 3. Compute scaled dot-product attention
    d_k = W_Q.size(1)
    scale = 1.0 / math.sqrt(d_k)
    
    # Compute attention scores: (seq_len,)
    scores = torch.matmul(K_new, q) * scale
    
    # Numerically stable softmax
    scores_max = torch.max(scores)
    exp_scores = torch.exp(scores - scores_max)
    weights = exp_scores / torch.sum(exp_scores)  # Shape: (seq_len,)
    
    # Compute the final attention output vector: shape (d_v,)
    output = torch.matmul(weights, V_new)
    
    # 4. Return output and updated cache tuple
    return output, (K_new, V_new)