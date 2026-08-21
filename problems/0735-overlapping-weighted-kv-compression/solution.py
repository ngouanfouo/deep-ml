import numpy as np

def overlapping_kv_compression(Z_a, Z_b, B_a, B_b, C_a, C_b):
    """
    Compute compressed KV entries from two overlapping score/value blocks.

    Args:
        Z_a: shape (N, M_a) - scores for block A
        Z_b: shape (N, M_b) - scores for block B
        B_a: shape (M_a,) - positional bias for block A
        B_b: shape (M_b,) - positional bias for block B
        C_a: shape (M_a, D) - values for block A
        C_b: shape (M_b, D) - values for block B

    Returns:
        Nested list of shape (N, D)
    """
    # Convert to numpy arrays
    Z_a = np.array(Z_a, dtype=np.float64)
    Z_b = np.array(Z_b, dtype=np.float64)
    B_a = np.array(B_a, dtype=np.float64)
    B_b = np.array(B_b, dtype=np.float64)
    C_a = np.array(C_a, dtype=np.float64)
    C_b = np.array(C_b, dtype=np.float64)
    
    N, M_a = Z_a.shape
    M_b = Z_b.shape[1]
    D = C_a.shape[1] if C_a.ndim > 1 else 1
    
    # Ensure C_a and C_b have proper shapes
    if C_a.ndim == 1:
        C_a = C_a.reshape(-1, 1)
    if C_b.ndim == 1:
        C_b = C_b.reshape(-1, 1)
    
    # Initialize output
    C_comp = np.zeros((N, D), dtype=np.float64)
    
    for i in range(N):
        # Add positional bias to score rows
        biased_scores_a = Z_a[i] + B_a  # shape (M_a,)
        biased_scores_b = Z_b[i] + B_b  # shape (M_b,)
        
        # Concatenate biased scores
        concatenated_scores = np.concatenate([biased_scores_a, biased_scores_b])  # shape (M_a + M_b,)
        
        # Apply numerically-stable softmax
        max_score = np.max(concatenated_scores)
        exp_scores = np.exp(concatenated_scores - max_score)
        softmax_weights = exp_scores / np.sum(exp_scores)
        
        # Split weights back
        W_a = softmax_weights[:M_a]
        W_b = softmax_weights[M_a:]
        
        # Produce compressed output
        C_comp[i] = W_a @ C_a + W_b @ C_b
    
    # Convert to nested list
    return C_comp.tolist()