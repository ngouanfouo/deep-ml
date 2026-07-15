import numpy as np

def sigmoid_moe_router(hidden_states, gate_weight, score_bias, top_k):
    """
    Implement sigmoid-based MoE routing with bias correction.
    """
    # Step 1: Compute router logits via matrix multiplication
    # hidden_states: (num_tokens, hidden_dim)
    # gate_weight: (num_experts, hidden_dim)
    # logits: (num_tokens, num_experts)
    logits = hidden_states @ gate_weight.T
    
    # Step 2: Apply sigmoid activation to get routing weights
    sigmoid_weights = 1 / (1 + np.exp(-logits))  # (num_tokens, num_experts)
    
    # Step 3: Add bias to determine expert selection
    # score_bias: (num_experts,)
    selection_scores = sigmoid_weights + score_bias  # (num_tokens, num_experts)
    
    # Step 4: Select top-k experts per token
    # Get indices of top-k experts based on selection_scores
    top_k_indices = np.argsort(-selection_scores, axis=1)[:, :top_k]  # (num_tokens, top_k)
    
    # Step 5: Gather the actual sigmoid weights (without bias) for selected experts
    # Use np.take_along_axis to gather weights for the selected indices
    top_k_weights = np.take_along_axis(sigmoid_weights, top_k_indices, axis=1)  # (num_tokens, top_k)
    
    # Step 6: Normalize the selected weights to sum to 1
    # Sum along the experts dimension for each token
    sum_weights = np.sum(top_k_weights, axis=1, keepdims=True)  # (num_tokens, 1)
    top_k_weights = top_k_weights / sum_weights  # (num_tokens, top_k)
    
    return top_k_weights, top_k_indices