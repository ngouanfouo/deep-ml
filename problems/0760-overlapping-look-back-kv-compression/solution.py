import numpy as np

def lookback_compressed_kv(H, W_aKV, W_bKV, W_aZ, W_bZ, B_a, B_b, m):
    """
    H: (n, d) hidden states; n must be divisible by m
    W_aKV, W_bKV: (d, c) KV projection weights (current, look-back)
    W_aZ, W_bZ: (d, c) compression-weight projection matrices
    B_a, B_b: (m, c) learnable positional biases
    m: block size

    Returns: list of lists of shape (n // m, c) with the compressed KV entries.
    """
    # Convert to numpy arrays
    H = np.array(H, dtype=np.float64)
    W_aKV = np.array(W_aKV, dtype=np.float64)
    W_bKV = np.array(W_bKV, dtype=np.float64)
    W_aZ = np.array(W_aZ, dtype=np.float64)
    W_bZ = np.array(W_bZ, dtype=np.float64)
    B_a = np.array(B_a, dtype=np.float64)
    B_b = np.array(B_b, dtype=np.float64)
    
    n, d = H.shape
    c = W_aKV.shape[1]
    num_blocks = n // m
    
    # Pre-compute KV entries for all tokens
    # Stream a KV: H @ W_aKV (n, c)
    KV_a = H @ W_aKV
    # Stream b KV: H @ W_bKV (n, c)
    KV_b = H @ W_bKV
    
    # Pre-compute logits for all tokens
    # Stream a logits: H @ W_aZ (n, c)
    Z_a = H @ W_aZ
    # Stream b logits: H @ W_bZ (n, c)
    Z_b = H @ W_bZ
    
    # Initialize compressed output
    compressed = np.zeros((num_blocks, c), dtype=np.float64)
    
    for i in range(num_blocks):
        # Current block indices: [m*i, m*(i+1))
        curr_start = m * i
        curr_end = m * (i + 1)
        
        # Stream a: current block tokens
        # KV entries: KV_a[curr_start:curr_end] (m, c)
        # Logits: Z_a[curr_start:curr_end] + B_a (m, c)
        logits_a = Z_a[curr_start:curr_end] + B_a  # (m, c)
        kv_a = KV_a[curr_start:curr_end]  # (m, c)
        
        # Stream b: look-back tokens (previous block)
        if i == 0:
            # First block: no previous block
            # Logits set to -inf, KV entries don't matter
            logits_b = np.full((m, c), -np.inf, dtype=np.float64)
            kv_b = np.zeros((m, c), dtype=np.float64)
        else:
            # Previous block indices: [m*(i-1), m*i)
            prev_start = m * (i - 1)
            prev_end = m * i
            logits_b = Z_b[prev_start:prev_end] + B_b  # (m, c)
            kv_b = KV_b[prev_start:prev_end]  # (m, c)
        
        # For each feature dimension independently
        for feat_idx in range(c):
            # Get logits for this feature: combine stream a and stream b
            all_logits = np.concatenate([logits_a[:, feat_idx], logits_b[:, feat_idx]])  # (2m,)
            
            # Stable softmax
            max_logit = np.max(all_logits)
            exp_logits = np.exp(all_logits - max_logit)
            weights = exp_logits / np.sum(exp_logits)
            
            # Combine KV entries
            all_kv = np.concatenate([kv_a[:, feat_idx], kv_b[:, feat_idx]])  # (2m,)
            compressed[i, feat_idx] = np.sum(weights * all_kv)
    
    return compressed.tolist()