import torch

def fuse_domain_experts(
    expert_scores: dict[str, dict[str, float]],
    domain_weights: dict[str, float],
    fusion_method: str
) -> float:
    """
    Fuse multiple domain expert models into a single score using PyTorch.
    
    Args:
        expert_scores: Dict of expert_name -> {domain: score}
        domain_weights: Dict of domain -> importance weight
        fusion_method: 'weighted_average' or 'best_per_domain'
    
    Returns:
        Overall fused score (weighted by domain importance) as a float
    """
    # Get all domains from domain_weights
    domains = list(domain_weights.keys())
    experts = list(expert_scores.keys())
    
    # Create tensors for scores
    # Shape: (n_experts, n_domains)
    scores_matrix = torch.zeros(len(experts), len(domains))
    
    for i, expert in enumerate(experts):
        for j, domain in enumerate(domains):
            scores_matrix[i, j] = expert_scores[expert].get(domain, 0.0)
    
    # Apply fusion method
    if fusion_method == 'weighted_average':
        # For each domain, compute weighted average of expert scores
        # Weights are based on how well each expert performs on that domain
        fused_domain_scores = torch.zeros(len(domains))
        
        for j in range(len(domains)):
            # Get scores for this domain from all experts
            domain_scores = scores_matrix[:, j]
            
            # Compute weights based on performance (normalized)
            # Better performance gets higher weight
            total_score = torch.sum(domain_scores)
            if total_score > 0:
                weights = domain_scores / total_score
            else:
                weights = torch.ones(len(experts)) / len(experts)
            
            # Weighted average
            fused_domain_scores[j] = torch.sum(weights * domain_scores)
    
    elif fusion_method == 'best_per_domain':
        # For each domain, select the maximum score among all experts
        fused_domain_scores, _ = torch.max(scores_matrix, dim=0)
    
    else:
        raise ValueError(f"Unknown fusion_method: {fusion_method}. Use 'weighted_average' or 'best_per_domain'.")
    
    # Convert domain weights to tensor
    weights_tensor = torch.tensor([domain_weights[domain] for domain in domains], dtype=torch.float32)
    
    # Compute overall weighted score
    overall_score = torch.sum(fused_domain_scores * weights_tensor)
    
    # Return as float
    return overall_score.item()