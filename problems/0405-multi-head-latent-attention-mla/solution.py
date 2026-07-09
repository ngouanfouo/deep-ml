import numpy as np

def multi_head_latent_attention(
    X: np.ndarray,
    W_dkv: np.ndarray,
    W_uk: np.ndarray,
    W_uv: np.ndarray,
    W_dq: np.ndarray,
    W_uq: np.ndarray,
    W_o: np.ndarray,
    n_heads: int
) -> tuple:
    """
    Perform Multi-Head Latent Attention (MLA).
    
    Args:
        X: Input tensor of shape (seq_len, d_model)
        W_dkv: Down-projection for KV compression (d_model, d_c_kv)
        W_uk: Up-projection for keys (d_c_kv, d_model)
        W_uv: Up-projection for values (d_c_kv, d_model)
        W_dq: Down-projection for query compression (d_model, d_c_q)
        W_uq: Up-projection for queries (d_c_q, d_model)
        W_o: Output projection (d_model, d_model)
        n_heads: Number of attention heads
    
    Returns:
        Tuple of (output, c_kv) where:
        - output: shape (seq_len, d_model)
        - c_kv: compressed KV latent of shape (seq_len, d_c_kv)
    """
    seq_len, d_model = X.shape
    d_head = d_model // n_heads
    
    # 1. Compress KV and reconstruct Keys and Values
    c_kv = X @ W_dkv               # (seq_len, d_c_kv)
    K = c_kv @ W_uk                # (seq_len, d_model)
    V = c_kv @ W_uv                # (seq_len, d_model)
    
    # 2. Compress Query and reconstruct Queries
    c_q = X @ W_dq                 # (seq_len, d_c_q)
    Q = c_q @ W_uq                 # (seq_len, d_model)
    
    # 3. Reshape and transpose for multi-head attention processing
    # Target shape for operations: (n_heads, seq_len, d_head)
    Q_heads = Q.reshape(seq_len, n_heads, d_head).transpose(1, 0, 2)
    K_heads = K.reshape(seq_len, n_heads, d_head).transpose(1, 0, 2)
    V_heads = V.reshape(seq_len, n_heads, d_head).transpose(1, 0, 2)
    
    # 4. Scaled dot-product attention per head
    # Scores shape: (n_heads, seq_len, seq_len)
    scores = (Q_heads @ K_heads.transpose(0, 2, 1)) / np.sqrt(d_head)
    
    # Numerically stable softmax along the last dimension
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    
    # Compute attention output per head: (n_heads, seq_len, d_head)
    context_heads = attn_weights @ V_heads
    
    # 5. Permute back and concatenate heads: (seq_len, d_model)
    context_concat = context_heads.transpose(1, 0, 2).reshape(seq_len, d_model)
    
    # 6. Apply final output projection
    output = context_concat @ W_o
    
    return output, c_kv