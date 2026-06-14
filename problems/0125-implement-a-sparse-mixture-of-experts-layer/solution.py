import numpy as np

def moe(x: np.ndarray, We: np.ndarray, Wg: np.ndarray, n_experts: int, top_k: int) -> np.ndarray:
    """
    Args:
        x: Input tensor of shape (n_batch, l_seq, d_model)
        We: Expert weights of shape (n_experts, d_model, d_model)
        Wg: Gating weights of shape (d_model, n_experts)
        n_experts: Number of experts
        top_k: Number of experts to route each token to
    Returns:
        Output tensor of shape (n_batch, l_seq, d_model)
    """
    n_batch, l_seq, d_model = x.shape
    
    # Compute gating scores: x @ Wg -> shape (n_batch, l_seq, n_experts)
    gate_scores = x @ Wg  # (n_batch, l_seq, n_experts)
    
    # Apply softmax to get probabilities
    # For numerical stability, subtract max
    gate_scores_stable = gate_scores - np.max(gate_scores, axis=-1, keepdims=True)
    exp_scores = np.exp(gate_scores_stable)
    gate_probs = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)  # (n_batch, l_seq, n_experts)
    
    # Top-k routing: select top_k experts per token
    # Get indices of top-k experts and their probabilities
    top_k_indices = np.argsort(-gate_probs, axis=-1)[:, :, :top_k]  # (n_batch, l_seq, top_k)
    top_k_probs = np.take_along_axis(gate_probs, top_k_indices, axis=-1)  # (n_batch, l_seq, top_k)
    
    # Normalize top-k probabilities to sum to 1
    normalized_probs = top_k_probs / np.sum(top_k_probs, axis=-1, keepdims=True)  # (n_batch, l_seq, top_k)
    
    # Initialize output tensor
    output = np.zeros_like(x)  # (n_batch, l_seq, d_model)
    
    # Apply experts and aggregate results
    for b in range(n_batch):
        for s in range(l_seq):
            token = x[b, s]  # (d_model,)
            expert_output = np.zeros(d_model)
            
            for k in range(top_k):
                expert_idx = top_k_indices[b, s, k]
                weight = normalized_probs[b, s, k]
                
                # Apply expert transformation
                expert_result = token @ We[expert_idx]  # (d_model,)
                
                # Add weighted contribution
                expert_output += weight * expert_result
            
            output[b, s] = expert_output
    
    return output