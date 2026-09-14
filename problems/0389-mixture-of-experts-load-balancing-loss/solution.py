import torch
import torch.nn.functional as F

def moe_load_balancing_loss(gate_logits: torch.Tensor, num_experts: int, alpha: float = 0.01) -> float:
    """
    Compute the load balancing auxiliary loss for a Mixture of Experts layer.
    
    Args:
        gate_logits: torch.Tensor of shape (num_tokens, num_experts), raw gating scores
        num_experts: int, number of experts
        alpha: float, scaling coefficient for the loss
    
    Returns:
        float: load balancing loss rounded to 4 decimal places
    """
    num_tokens = gate_logits.size(0)
    
    # 1. Convert raw gate logits into routing probabilities per token
    routing_probs = F.softmax(gate_logits, dim=-1)  # Shape: (num_tokens, num_experts)
    
    # 2. Determine hard routing assignments using argmax
    expert_assignments = torch.argmax(routing_probs, dim=-1)  # Shape: (num_tokens,)
    
    # 3. Compute the dispatch fraction for each expert
    # Use one_hot to count assignments and normalize by total tokens
    dispatch_counts = F.one_hot(expert_assignments, num_classes=num_experts).sum(dim=0).float()
    dispatch_fraction = dispatch_counts / num_tokens  # Shape: (num_experts,)
    
    # 4. Compute the average gating probability for each expert
    mean_probs = routing_probs.mean(dim=0)  # Shape: (num_experts,)
    
    # 5. Compute the load balancing loss: alpha * N * sum(f_i * p_i)
    loss = alpha * num_experts * torch.sum(dispatch_fraction * mean_probs)
    
    return round(loss.item(), 4)