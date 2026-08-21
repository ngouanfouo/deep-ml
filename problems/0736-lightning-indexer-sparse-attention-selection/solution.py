import numpy as np

def lightning_indexer_topk(H_q, K_comp, W_down, W_up, w_head, k):
    """
    Select top-k most relevant compressed key-value entries for each query token.
    
    Args:
        H_q: shape (n_q, d) - query hidden states
        K_comp: shape (n_kv, c) - compressed indexer keys
        W_down: shape (d, r) - down-projection matrix
        W_up: shape (r, h*c) - up-projection matrix
        w_head: length h - per-head scalar weights
        k: number of top entries to select
        
    Returns:
        Nested list of shape (n_q, min(k, n_kv)) of integer indices
    """
    # Convert to numpy arrays
    H_q = np.array(H_q, dtype=np.float64)
    K_comp = np.array(K_comp, dtype=np.float64)
    W_down = np.array(W_down, dtype=np.float64)
    W_up = np.array(W_up, dtype=np.float64)
    w_head = np.array(w_head, dtype=np.float64)
    
    n_q, d = H_q.shape
    n_kv, c = K_comp.shape
    h = len(w_head)
    
    # Calculate r from W_down shape
    r = W_down.shape[1]
    
    # Check that W_up has correct shape
    h_c = W_up.shape[1]
    # h*c should equal h_c, so c = h_c // h
    c_from_up = h_c // h
    
    # Ensure c matches
    assert c_from_up == c, f"c from W_up ({c_from_up}) doesn't match K_comp dimension ({c})"
    
    # If k > n_kv, use n_kv
    actual_k = min(k, n_kv)
    
    # Initialize result
    result = []
    
    for i in range(n_q):
        # Project query through bottleneck: first down, then up
        query_down = H_q[i] @ W_down  # shape (r,)
        query_up = query_down @ W_up   # shape (h*c,)
        
        # Reshape into h indexer-query heads, each of dimension c
        query_heads = query_up.reshape(h, c)  # shape (h, c)
        
        # Compute combined scores for each compressed key
        combined_scores = np.zeros(n_kv, dtype=np.float64)
        
        for j in range(n_kv):
            # For each compressed key, compute dot product with each head
            scores_per_head = np.zeros(h, dtype=np.float64)
            for head_idx in range(h):
                # Dot product between head's indexer query and compressed key
                score = np.dot(query_heads[head_idx], K_comp[j])
                # Apply ReLU
                score = max(0, score)
                scores_per_head[head_idx] = score
            
            # Weighted sum across heads
            combined_scores[j] = np.sum(w_head * scores_per_head)
        
        # Get top-k indices by descending score, tie-breaking by smaller index
        # Sort indices by (-score, index)
        sorted_indices = np.argsort([-combined_scores[idx] for idx in range(n_kv)])
        
        # Select top-k
        top_indices = sorted_indices[:actual_k].tolist()
        
        result.append(top_indices)
    
    return result