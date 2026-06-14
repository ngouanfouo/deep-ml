import torch
import torch.nn.functional as F
from typing import Tuple

def compute_qkv(X: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Query, Key, and Value matrices.

    Args:
        X: Input matrix of shape (seq_len, d_model)
        W_q, W_k, W_v: Weight matrices of shape (d_model, d_model)

    Returns:
        Q, K, V matrices each of shape (seq_len, d_model)
    """
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    return Q, K, V

def self_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product self-attention.

    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_k)

    Returns:
        Attention output of shape (seq_len, d_k)
    """
    seq_len, d_k = Q.shape
    
    # Compute scaled dot-product attention scores
    scores = Q @ K.T / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    
    # Apply numerically stable softmax (subtract max before exponentiating)
    scores_max = scores.max(dim=-1, keepdim=True)[0]
    exp_scores = torch.exp(scores - scores_max)
    attention_weights = exp_scores / exp_scores.sum(dim=-1, keepdim=True)
    
    # Apply attention weights to values
    output = attention_weights @ V
    
    return output

def multi_head_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, n_heads: int) -> torch.Tensor:
    """
    Compute multi-head attention.

    Args:
        Q, K, V: Matrices of shape (seq_len, d_model)
        n_heads: Number of attention heads

    Returns:
        Attention output of shape (seq_len, d_model)
    """
    seq_len, d_model = Q.shape
    head_dim = d_model // n_heads
    
    # Reshape Q, K, V to split into multiple heads
    # Shape: (seq_len, n_heads, head_dim)
    Q = Q.reshape(seq_len, n_heads, head_dim)
    K = K.reshape(seq_len, n_heads, head_dim)
    V = V.reshape(seq_len, n_heads, head_dim)
    
    # Transpose to (n_heads, seq_len, head_dim) for independent attention computation
    Q = Q.transpose(0, 1)
    K = K.transpose(0, 1)
    V = V.transpose(0, 1)
    
    # Compute attention for each head
    # Initialize output tensor
    head_outputs = []
    for h in range(n_heads):
        head_output = self_attention(Q[h], K[h], V[h])  # (seq_len, head_dim)
        head_outputs.append(head_output)
    
    # Concatenate heads along feature dimension
    # Stack: (n_heads, seq_len, head_dim) -> transpose -> (seq_len, n_heads, head_dim) -> reshape
    concatenated = torch.stack(head_outputs, dim=0)  # (n_heads, seq_len, head_dim)
    concatenated = concatenated.transpose(0, 1)  # (seq_len, n_heads, head_dim)
    output = concatenated.reshape(seq_len, d_model)  # (seq_len, d_model)
    
    return output