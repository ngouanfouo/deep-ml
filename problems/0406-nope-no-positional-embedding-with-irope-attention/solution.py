import numpy as np

def apply_rope(X: np.ndarray, positions: np.ndarray, base: float) -> np.ndarray:
    """
    Applies Rotary Position Embeddings (RoPE) to the input matrix.
    Rotates consecutive pairs of dimensions (2i, 2i+1) by theta = pos / base**(2i/d_head).
    """
    seq_len, d_head = X.shape
    X_rotated = np.zeros_like(X)
    
    # Calculate frequencies for each independent pair of dimensions
    # there are d_head // 2 pairs
    dim_pairs = d_head // 2
    inv_freq = 1.0 / (base ** (np.arange(0, d_head, 2) / d_head))
    
    # Compute angles shape: (seq_len, dim_pairs)
    angles = np.outer(positions, inv_freq)
    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)
    
    # Apply rotation sequentially to each interleaved (2i, 2i+1) pair
    for i in range(dim_pairs):
        x0 = X[:, 2 * i]
        x1 = X[:, 2 * i + 1]
        
        cos = cos_angles[:, i]
        sin = sin_angles[:, i]
        
        X_rotated[:, 2 * i] = x0 * cos - x1 * sin
        X_rotated[:, 2 * i + 1] = x1 * cos + x0 * sin
        
    return X_rotated

def irope_attention(Q: list, K: list, V: list, positions: list, layer_index: int, rope_layers: list, base: float = 10000.0) -> dict:
    """
    Compute attention for a single layer in an iRoPE transformer.
    
    Args:
        Q: Query matrix, shape (seq_len, d_head)
        K: Key matrix, shape (seq_len, d_head)
        V: Value matrix, shape (seq_len, d_head)
        positions: Position indices, shape (seq_len,)
        layer_index: Current layer index (0-indexed)
        rope_layers: List of layer indices that use RoPE
        base: Base frequency for RoPE
    
    Returns:
        Dictionary with 'output', 'attention_weights', and 'uses_rope'
    """
    # Convert input structures to NumPy arrays
    Q_arr = np.array(Q, dtype=float)
    K_arr = np.array(K, dtype=float)
    V_arr = np.array(V, dtype=float)
    pos_arr = np.array(positions, dtype=int)
    
    seq_len, d_head = Q_arr.shape
    
    # 1. Check whether RoPE or NoPE is triggered
    uses_rope = layer_index in rope_layers
    
    # 2. Modify Q and K if RoPE is enabled
    if uses_rope:
        Q_out = apply_rope(Q_arr, pos_arr, base)
        K_out = apply_rope(K_arr, pos_arr, base)
    else:
        Q_out = Q_arr
        K_out = K_arr
        
    # 3. Compute scaled dot-product attention
    # Attention scores shape: (seq_len, seq_len)
    scores = (Q_out @ K_out.T) / np.sqrt(d_head)
    
    # Numerically stable softmax
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    
    # Compute output context representation
    output = attention_weights @ V_arr
    
    # 4. Format and round outputs up to 4 decimal places
    return {
        'output': np.round(output, 4).tolist(),
        'attention_weights': np.round(attention_weights, 4).tolist(),
        'uses_rope': uses_rope
    }