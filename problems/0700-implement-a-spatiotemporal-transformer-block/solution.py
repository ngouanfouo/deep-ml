import numpy as np

def layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Standard Layer Normalization over the last feature dimension (D)."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax along a specified axis."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def spatiotemporal_transformer_block(X: np.ndarray, params: dict) -> np.ndarray:
    """
    Forward pass of a spatiotemporal transformer block.
    
    Args:
        X: Input tensor of shape (T, N, D).
        params: Dict with attention and FFN weight matrices.
    
    Returns:
        Output tensor of shape (T, N, D).
    """
    T, N, D = X.shape
    scale = np.sqrt(D)
    
    # -------------------------------------------------------------------------
    # 1. Spatial Self-Attention Sub-layer
    #    Operates independently for each frame t across spatial tokens N.
    # -------------------------------------------------------------------------
    X_norm = layer_norm(X)
    
    Q_s = X_norm @ params['Wqs']  # (T, N, D)
    K_s = X_norm @ params['Wks']  # (T, N, D)
    V_s = X_norm @ params['Wvs']  # (T, N, D)
    
    # Scores shape: (T, N, N)
    scores_s = np.matmul(Q_s, K_s.transpose(0, 2, 1)) / scale
    attn_s = softmax(scores_s, axis=-1)
    
    context_s = np.matmul(attn_s, V_s)  # (T, N, D)
    out_s = context_s @ params['Wos']    # (T, N, D)
    
    X = X + out_s  # Residual connection
    
    # -------------------------------------------------------------------------
    # 2. Temporal Self-Attention Sub-layer
    #    Operates independently for each spatial position n across time steps T.
    # -------------------------------------------------------------------------
    X_norm = layer_norm(X)
    
    # Transpose to shape (N, T, D) to process each spatial position independently
    Q_t = (X_norm @ params['Wqt']).transpose(1, 0, 2)
    K_t = (X_norm @ params['Wkt']).transpose(1, 0, 2)
    V_t = (X_norm @ params['Wvt']).transpose(1, 0, 2)
    
    # Scores shape: (N, T, T)
    scores_t = np.matmul(Q_t, K_t.transpose(0, 2, 1)) / scale
    attn_t = softmax(scores_t, axis=-1)
    
    context_t = np.matmul(attn_t, V_t)                     # (N, T, D)
    out_t = (context_t @ params['Wot']).transpose(1, 0, 2)  # Back to (T, N, D)
    
    X = X + out_t  # Residual connection
    
    # -------------------------------------------------------------------------
    # 3. Position-wise Feed-Forward Network (FFN) Sub-layer
    # -------------------------------------------------------------------------
    X_norm = layer_norm(X)
    
    ffn_hidden = np.maximum(0, X_norm @ params['W1'])  # ReLU activation
    out_ffn = ffn_hidden @ params['W2']                # Second projection
    
    X = X + out_ffn  # Residual connection
    
    return X