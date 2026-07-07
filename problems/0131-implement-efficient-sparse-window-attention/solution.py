import numpy as np

def sparse_window_attention(Q, K, V, window_size, scale_factor=None):
    """
    Computes sparse attention over a sequence using a sliding fixed-radius window.
    
    Args:
        Q: Query array of shape (seq_len, d_k)
        K: Key array of shape (seq_len, d_k)
        V: Value array of shape (seq_len, d_v)
        window_size: Integer radius (w) of the sliding window
        scale_factor: Scaling factor for dot-product scores (defaults to 1 / sqrt(d_k))
        
    Returns:
        Output array of shape (seq_len, d_v)
    """
    seq_len, d_k = Q.shape
    _, d_v = V.shape
    
    # 1. Set default scaling factor if not provided
    if scale_factor is None:
        scale_factor = 1.0 / np.sqrt(d_k)
        
    # Initialize the output matrix
    output = np.zeros((seq_len, d_v))
    
    # 2. Iterate through each token in the sequence
    for i in range(seq_len):
        # Calculate the localized window boundaries (inclusive boundaries handled via slicing)
        start_idx = max(0, i - window_size)
        end_idx = min(seq_len, i + window_size + 1)  # +1 because Python slicing is exclusive
        
        # Extract local keys and values within the active window
        K_local = K[start_idx:end_idx]  # Shape: (window_elements, d_k)
        V_local = V[start_idx:end_idx]  # Shape: (window_elements, d_v)
        
        # Current query token vector
        q_i = Q[i]  # Shape: (d_k,)
        
        # 3. Compute raw dot-product attention scores for the window
        # Matrix multiplication / dot product between vector and local window matrix
        scores = np.dot(K_local, q_i) * scale_factor  # Shape: (window_elements,)
        
        # 4. Numerically stable softmax over the active window scores
        max_score = np.max(scores)
        exp_scores = np.exp(scores - max_score)
        attention_weights = exp_scores / np.sum(exp_scores)
        
        # 5. Aggregate local values using the attention weights
        # Shape: (d_v,)
        output[i] = np.dot(attention_weights, V_local)
        
    return output