import torch
import torch.nn.functional as F

def compute_ptx_loss_pytorch(
    rl_loss: torch.Tensor,
    pretrain_logits: torch.Tensor,
    pretrain_labels: torch.Tensor,
    beta_ptx: float = 0.1
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute PTX Loss using PyTorch.
    
    Args:
        rl_loss: RL loss tensor (scalar)
        pretrain_logits: Shape (batch_size, vocab_size)
        pretrain_labels: Shape (batch_size,)
        beta_ptx: Weight coefficient
    
    Returns:
        (total_loss, ce_loss, weighted_ce_loss) as tensors
        
    Hints:
        - Use F.cross_entropy() for numerical stability
        - cross_entropy automatically applies softmax
        - reduction='mean' for batch averaging
    """
    # Compute cross-entropy loss on pre-training batch
    # F.cross_entropy applies log_softmax internally and computes NLL loss
    ce_loss = F.cross_entropy(pretrain_logits, pretrain_labels, reduction='mean')
    
    # Apply beta coefficient to get weighted PTX component
    weighted_ce_loss = beta_ptx * ce_loss
    
    # Total loss = RL loss + PTX loss
    total_loss = rl_loss + weighted_ce_loss
    
    return (total_loss, ce_loss, weighted_ce_loss)