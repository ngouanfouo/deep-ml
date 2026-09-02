import numpy as np

def cross_layer_kv_attention(x, layer_weights, n_kv_producing_layers):
    """
    Multi-layer causal self-attention with cross-layer KV sharing.

    Args:
        x: list of lists, shape (seq_len, d_model)
        layer_weights: list of dicts, each containing 'Wq', 'Wk', 'Wv'
                    (each d_model x d_model)
        n_kv_producing_layers: int, number of initial layers that compute
                            their own K and V; later layers reuse the
                            most recent shared K, V.

    Returns:
        Final hidden state as a nested list, rounded to 4 decimal places.
    """
    h = np.array(x, dtype=float)
    seq_len, d_model = h.shape
    
    shared_K = None
    shared_V = None
    
    # Create causal mask: upper triangle above the main diagonal gets -1e9, else 0
    # Shape: (seq_len, seq_len)
    mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9
    
    for i, weights in enumerate(layer_weights):
        Wq = np.array(weights['Wq'], dtype=float)
        Wk = np.array(weights['Wk'], dtype=float)
        Wv = np.array(weights['Wv'], dtype=float)
        
        # Compute Q from the current hidden state
        Q = h @ Wq
        
        # Compute or reuse K and V
        if i < n_kv_producing_layers:
            shared_K = h @ Wk
            shared_V = h @ Wv
        
        # Scaled dot-product attention
        scores = (Q @ shared_K.T) / np.sqrt(d_model) + mask
        
        # Numerically stable softmax (subtract row-wise max before exponentiating)
        scores_max = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        
        # Compute attention output
        attn = attn_weights @ shared_V
        
        # Apply residual connection
        h = h + attn
        
    return np.round(h, 4).tolist()