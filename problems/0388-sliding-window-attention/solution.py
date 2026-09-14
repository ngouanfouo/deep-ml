import torch
import math

def sliding_window_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Compute sliding window attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_v)
        window_size: Number of positions to the left and right each query can attend to
    
    Returns:
        Output matrix of shape (seq_len, d_v), rounded to 4 decimal places.
    """
    seq_len, d_k = Q.shape
    
    # 1. Compute scaled dot-product attention scores: (seq_len, seq_len)
    scale = 1.0 / math.sqrt(d_k)
    scores = torch.matmul(Q, K.T) * scale
    
    # 2. Create sliding window mask where |i - j| > window_size
    indices = torch.arange(seq_len, device=Q.device)
    diff = torch.abs(indices.unsqueeze(0) - indices.unsqueeze(1))
    mask = diff > window_size
    
    # Apply mask by filling out-of-window positions with negative infinity
    scores.masked_fill_(mask, -float('inf'))
    
    # 3. Apply numerically stable softmax row-wise (PyTorch handles -inf correctly)
    weights = torch.softmax(scores, dim=-1)
    
    # 4. Compute the output by multiplying attention weights with V
    output = torch.matmul(weights, V)
    
    # 5. Round output to 4 decimal places
    return torch.round(output, decimals=4)