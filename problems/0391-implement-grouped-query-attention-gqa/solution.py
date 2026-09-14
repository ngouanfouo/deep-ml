import torch
import math

def grouped_query_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, num_heads: int, num_kv_heads: int) -> torch.Tensor:
    """
    Compute Grouped Query Attention.

    Args:
        Q: Query tensor, shape (batch_size, seq_len, num_heads * head_dim)
        K: Key tensor, shape (batch_size, seq_len, num_kv_heads * head_dim)
        V: Value tensor, shape (batch_size, seq_len, num_kv_heads * head_dim)
        num_heads: Number of query heads
        num_kv_heads: Number of key/value heads

    Returns:
        Output tensor, shape (batch_size, seq_len, num_heads * head_dim)
    """
    if num_heads % num_kv_heads != 0:
        raise ValueError("num_heads must be evenly divisible by num_kv_heads.")
        
    batch_size, seq_len, _ = Q.shape
    head_dim = Q.shape[-1] // num_heads
    
    # 1. Reshape and transpose Q, K, V to have explicit head dimensions:
    # Q: (batch_size, num_heads, seq_len, head_dim)
    Q = Q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    
    # K, V: (batch_size, num_kv_heads, seq_len, head_dim)
    K = K.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    V = V.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    
    # 2. Compute the number of query heads per KV group
    num_queries_per_kv = num_heads // num_kv_heads
    
    # 3. If num_kv_heads < num_heads, expand/repeat K and V to match num_heads
    if num_queries_per_kv > 1:
        # Using repeat_interleave so each KV head is shared across its group of query heads
        K = K.repeat_interleave(num_queries_per_kv, dim=1)
        V = V.repeat_interleave(num_queries_per_kv, dim=1)
        
    # Now K and V have shape (batch_size, num_heads, seq_len, head_dim)
    
    # 4. Compute scaled dot-product attention
    scale = 1.0 / math.sqrt(head_dim)
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # (batch_size, num_heads, seq_len, seq_len)
    
    # Apply numerically stable softmax row-wise
    attn_weights = torch.softmax(scores, dim=-1)
    
    # Compute attention output per head: (batch_size, num_heads, seq_len, head_dim)
    attn_output = torch.matmul(attn_weights, V)
    
    # 5. Transpose and flatten back to original shape: (batch_size, seq_len, num_heads * head_dim)
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, num_heads * head_dim)
    
    return attn_output