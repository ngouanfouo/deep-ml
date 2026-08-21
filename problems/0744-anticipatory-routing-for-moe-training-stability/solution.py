import numpy as np

def anticipatory_moe(features, router_weights, expert_weights, delta):
    """
    Top-1 MoE forward pass with decoupled (anticipatory) routing.

    Args:
        features: list of T arrays, each of shape (N, d)
        router_weights: list of T arrays, each of shape (d, E)
        expert_weights: list of T arrays, each of shape (E, d, d_out)
        delta: non-negative integer, routing lookback offset

    Returns:
        List of T outputs, each a nested list of shape (N, d_out).
    """
    # Convert all inputs to numpy arrays
    features = [np.array(f, dtype=np.float64) for f in features]
    router_weights = [np.array(r, dtype=np.float64) for r in router_weights]
    expert_weights = [[np.array(e, dtype=np.float64) for e in expert_set] for expert_set in expert_weights]
    
    T = len(features)
    outputs = []
    
    for t in range(T):
        # Step 1: Select the router snapshot
        if t >= delta:
            W_r = router_weights[t - delta]
        else:
            W_r = router_weights[t]
        
        # Step 2: Compute routing scores
        scores = features[t] @ W_r  # shape (N, E)
        
        # Step 3: Pick top-1 expert index for each token
        idx = np.argmax(scores, axis=1)  # shape (N,)
        
        # Step 4: Compute per-token outputs using current experts
        N, d = features[t].shape
        d_out = expert_weights[t][0].shape[1]  # Get d_out from first expert
        step_outputs = np.zeros((N, d_out), dtype=np.float64)
        
        for i in range(N):
            expert_idx = idx[i]
            step_outputs[i] = features[t][i] @ expert_weights[t][expert_idx]
        
        # Convert to nested list
        outputs.append(step_outputs.tolist())
    
    return outputs