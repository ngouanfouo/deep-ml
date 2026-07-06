import numpy as np

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray):
    """
    Compute Query (Q), Key (K), and Value (V) matrices.
    """
    return np.dot(X, W_q), np.dot(X, W_k), np.dot(X, W_v)

def masked_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Compute masked self-attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_model)
        K: Key matrix of shape (seq_len, d_model)
        V: Value matrix of shape (seq_len, d_model)
        mask: Mask matrix of shape (seq_len, seq_len) with 0s and -inf
        
    Returns:
        np.ndarray: Masked attention output of shape (seq_len, d_model)
    """
    d_model = Q.shape[-1]
    
    # Step 1: Calculate raw attention scores via dot product
    scores = np.dot(Q, K.T)
    
    # Step 2: Scale the scores by the square root of d_model
    scaled_scores = scores / np.sqrt(d_model)
    
    # Step 3: Apply the mask (adds 0 to allowed positions, -inf to blocked ones)
    masked_scores = scaled_scores + mask
    
    # Step 4: Softmax stability trick: subtract max row value before exponentiating
    max_scores = np.max(masked_scores, axis=-1, keepdims=True)
    exp_scores = np.exp(masked_scores - max_scores)
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    
    # Step 5: Compute weighted sum over the Value matrix
    output = np.dot(attention_weights, V)
    
    return output