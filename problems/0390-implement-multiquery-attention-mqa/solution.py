import torch
import math

def multiquery_attention(X: torch.Tensor, W_queries: list, W_key: torch.Tensor, W_value: torch.Tensor, W_out: torch.Tensor) -> torch.Tensor:
    """
    Compute Multi-Query Attention.
    
    Args:
        X: Input tensor of shape (seq_len, d_model)
        W_queries: List of query weight tensors, each (d_model, d_k), one per head
        W_key: Shared key weight tensor of shape (d_model, d_k)
        W_value: Shared value weight tensor of shape (d_model, d_v)
        W_out: Output projection tensor of shape (num_heads * d_v, d_model)
    
    Returns:
        Output tensor of shape (seq_len, d_model), rounded to 4 decimal places
    """
    # 1. Compute the shared Key and Value matrices across all heads
    K = torch.matmul(X, W_key)  # Shape: (seq_len, d_k)
    V = torch.matmul(X, W_value)  # Shape: (seq_len, d_v)
    
    d_k = K.size(-1)
    scale = 1.0 / math.sqrt(d_k)
    
    head_outputs = []
    
    # 2. Iterate through each head's query projection
    for W_Q in W_queries:
        Q = torch.matmul(X, W_Q)  # Shape: (seq_len, d_k)
        
        # Compute scaled dot-product attention scores: (seq_len, seq_len)
        scores = torch.matmul(Q, K.T) * scale
        
        # Apply numerically stable softmax row-wise
        weights = torch.softmax(scores, dim=-1)
        
        # Compute head output: (seq_len, d_v)
        head_out = torch.matmul(weights, V)
        head_outputs.append(head_out)
        
    # 3. Concatenate all head outputs along the feature dimension: (seq_len, num_heads * d_v)
    concat_out = torch.cat(head_outputs, dim=-1)
    
    # 4. Apply the final output projection: (seq_len, d_model)
    output = torch.matmul(concat_out, W_out)
    
    # 5. Round final output tensor to 4 decimal places
    return torch.round(output, decimals=4)