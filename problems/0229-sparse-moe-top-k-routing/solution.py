import numpy as np

def moe_topk_routing(
    router_logits: np.ndarray,
    expert_outputs: np.ndarray,
    k: int
) -> np.ndarray:
    """
    Perform top-k expert routing for a Mixture-of-Experts layer.
    
    For each token:
    1. Select the top-k experts based on router_logits
    2. Compute softmax weights over only the selected experts
    3. Return weighted combination of the selected expert outputs
    
    Args:
        router_logits: Shape (batch_size, num_experts)
                      Raw scores from the router for each expert
        expert_outputs: Shape (batch_size, num_experts, hidden_dim)
                       Output from each expert for each input
        k: Number of experts to select per token
        
    Returns:
        Shape (batch_size, hidden_dim) - weighted combination of expert outputs
    """
    # Convert to numpy arrays if needed
    router_logits = np.array(router_logits, dtype=np.float64)
    expert_outputs = np.array(expert_outputs, dtype=np.float64)
    
    batch_size, num_experts = router_logits.shape
    hidden_dim = expert_outputs.shape[2]
    
    # Initialize output array
    output = np.zeros((batch_size, hidden_dim), dtype=np.float64)
    
    for b in range(batch_size):
        # Get router logits for this batch item
        logits = router_logits[b]
        
        # Find top-k expert indices
        top_k_indices = np.argsort(logits)[-k:][::-1]  # Descending order
        
        # Get logits for top-k experts
        top_k_logits = logits[top_k_indices]
        
        # Compute softmax weights over top-k experts
        # For numerical stability, subtract max before exp
        top_k_logits_shifted = top_k_logits - np.max(top_k_logits)
        exp_logits = np.exp(top_k_logits_shifted)
        weights = exp_logits / np.sum(exp_logits)
        
        # Get expert outputs for top-k experts
        top_k_outputs = expert_outputs[b, top_k_indices]  # Shape: (k, hidden_dim)
        
        # Compute weighted combination
        # output[b] = sum(weights[i] * top_k_outputs[i] for i in range(k))
        weighted_output = np.sum(weights[:, np.newaxis] * top_k_outputs, axis=0)
        output[b] = weighted_output
    
    return output