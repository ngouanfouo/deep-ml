import torch
import torch.nn.functional as F

def contrastive_loss(embeddings: torch.Tensor, temperature: float) -> float:
    """
    Compute the NT-Xent (SimCLR-style) contrastive loss.
    
    Args:
        embeddings: Tensor of shape (2N, d) where consecutive pairs
                    (2i, 2i+1) are positive pairs.
        temperature: Temperature scaling parameter (tau > 0).
    
    Returns:
        The mean contrastive loss as a float.
    """
    batch_size = embeddings.size(0)
    
    # 1. L2 normalize the embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    # 2. Compute pairwise cosine similarities and scale by temperature
    sim_matrix = torch.matmul(embeddings, embeddings.T) / temperature  # Shape: (2N, 2N)
    
    # 3. Identify positive pair indices: 0 <-> 1, 2 <-> 3, etc. (bitwise XOR with 1)
    pos_indices = torch.arange(batch_size, device=embeddings.device) ^ 1
    pos_sim = sim_matrix[torch.arange(batch_size), pos_indices]  # Shape: (2N,)
    
    # 4. Mask out self-similarities (diagonal elements) for the denominator using -inf
    mask = torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
    sim_matrix.masked_fill_(mask, -float('inf'))
    
    # 5. Compute log-sum-exp over all negative and positive samples (excluding self)
    log_sum_exp = torch.logsumexp(sim_matrix, dim=1)  # Shape: (2N,)
    
    # 6. Compute individual anchor losses: -pos_sim + log_sum_exp
    anchor_losses = log_sum_exp - pos_sim
    
    # 7. Return the mean loss across all anchors as a Python float
    return anchor_losses.mean().item()