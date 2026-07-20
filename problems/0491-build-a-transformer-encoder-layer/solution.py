import numpy as np

def transformer_encoder_layer(X: np.ndarray, weights: dict, num_heads: int, eps: float = 1e-5) -> np.ndarray:
    """
    Forward pass of a single Transformer Encoder Layer using only NumPy.

    Args:
        X: Input tensor of shape (batch_size, seq_len, d_model)
        weights: Dictionary containing all weight matrices and normalization parameters
        num_heads: Number of attention heads
        eps: Epsilon for layer normalization

    Returns:
        Output tensor of shape (batch_size, seq_len, d_model)
    """
    batch_size, seq_len, d_model = X.shape
    d_k = d_model // num_heads

    # Helper function for Layer Normalization
    def layer_norm(x, gamma, beta):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return gamma * (x - mean) / np.sqrt(var + eps) + beta

    # --- 1. Multi-Head Self-Attention ---
    # Linear projections: (batch_size, seq_len, d_model)
    Q = np.matmul(X, weights['W_q'])
    K = np.matmul(X, weights['W_k'])
    V = np.matmul(X, weights['W_v'])

    # Reshape and transpose to split heads: (batch_size, num_heads, seq_len, d_k)
    Q_heads = Q.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    K_heads = K.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    V_heads = V.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)

    # Scaled dot-product attention scores: (batch_size, num_heads, seq_len, seq_len)
    scores = np.matmul(Q_heads, K_heads.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
    
    # Numerically stable softmax
    scores_max = np.max(scores, axis=-1, keepdims=True)
    attn_weights = np.exp(scores - scores_max)
    attn_weights /= np.sum(attn_weights, axis=-1, keepdims=True)

    # Attention context: (batch_size, num_heads, seq_len, d_k)
    context_heads = np.matmul(attn_weights, V_heads)

    # Concatenate heads: (batch_size, seq_len, d_model)
    context = context_heads.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
    
    # Final attention output projection
    attn_out = np.matmul(context, weights['W_o'])

    # --- 2. First Add & Norm ---
    x_norm1 = layer_norm(X + attn_out, weights['gamma1'], weights['beta1'])

    # --- 3. Position-wise Feed-Forward Network ---
    ffn_out = np.maximum(0, np.matmul(x_norm1, weights['W1']) + weights['b1']) # ReLU
    ffn_out = np.matmul(ffn_out, weights['W2']) + weights['b2']

    # --- 4. Second Add & Norm ---
    out = layer_norm(x_norm1 + ffn_out, weights['gamma2'], weights['beta2'])

    return out