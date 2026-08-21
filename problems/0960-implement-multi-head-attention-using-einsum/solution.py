import numpy as np

def multi_head_attention_einsum(X, W_q, W_k, W_v, W_o, num_heads: int, causal: bool = True):
    """
    Multi-head self-attention computed with numpy.einsum.

    Args:
        X: input array of shape (B, S, d_in)
        W_q, W_k, W_v: projection matrices of shape (d_in, d_out)
        W_o: output projection matrix of shape (d_out, d_out)
        num_heads: number of attention heads (must divide d_out)
        causal: if True, apply a causal mask so position t only attends to <= t

    Returns:
        Nested list of shape (B, S, d_out) representing the attention output.
    """
    # Convert inputs to numpy arrays
    X = np.array(X, dtype=np.float64)
    W_q = np.array(W_q, dtype=np.float64)
    W_k = np.array(W_k, dtype=np.float64)
    W_v = np.array(W_v, dtype=np.float64)
    W_o = np.array(W_o, dtype=np.float64)
    
    B, S, d_in = X.shape
    d_out = W_q.shape[1]
    
    # Assert d_out is divisible by num_heads
    assert d_out % num_heads == 0, f"d_out ({d_out}) must be divisible by num_heads ({num_heads})"
    d_head = d_out // num_heads
    
    # 1. Project X to Q, K, V using einsum
    # X: (B, S, d_in), W: (d_in, d_out) -> (B, S, d_out)
    Q = np.einsum('bsi,io->bso', X, W_q)
    K = np.einsum('bsi,io->bso', X, W_k)
    V = np.einsum('bsi,io->bso', X, W_v)
    
    # 2. Reshape into heads: (B, S, num_heads, d_head) -> (B, num_heads, S, d_head)
    Q = Q.reshape(B, S, num_heads, d_head).transpose(0, 2, 1, 3)  # (B, num_heads, S, d_head)
    K = K.reshape(B, S, num_heads, d_head).transpose(0, 2, 1, 3)  # (B, num_heads, S, d_head)
    V = V.reshape(B, S, num_heads, d_head).transpose(0, 2, 1, 3)  # (B, num_heads, S, d_head)
    
    # 3. Compute per-head attention scores: Q @ K^T / sqrt(d_head)
    # Q: (B, num_heads, S, d_head), K: (B, num_heads, S, d_head)
    # scores: (B, num_heads, S, S)
    scores = np.einsum('bhqd,bhkd->bhqk', Q, K) / np.sqrt(d_head)
    
    # 4. Apply causal masking if requested
    if causal:
        # Create causal mask: upper triangular matrix of -inf
        mask = np.triu(np.ones((S, S), dtype=np.float64), k=1)
        mask = np.where(mask == 1, -np.inf, 0.0)
        # Add mask to scores (broadcast over batch and head dimensions)
        scores = scores + mask.reshape(1, 1, S, S)
    
    # 5. Softmax along the key axis (last dimension)
    # Subtract max for numerical stability
    max_vals = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - max_vals)
    attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    
    # 6. Aggregate values: attn_weights @ V
    # attn_weights: (B, num_heads, S, S), V: (B, num_heads, S, d_head)
    # context: (B, num_heads, S, d_head)
    context = np.einsum('bhqk,bhkd->bhqd', attn_weights, V)
    
    # 7. Reshape back to (B, S, d_out)
    context = context.transpose(0, 2, 1, 3).reshape(B, S, d_out)
    
    # 8. Apply output projection
    output = np.einsum('bsi,io->bso', context, W_o)
    
    # Return as nested list
    return output.tolist()