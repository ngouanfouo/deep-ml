import torch
import torch.nn.functional as F

def smooth_labels(y_true: torch.Tensor, num_classes: int, epsilon: float) -> torch.Tensor:
    """
    Create smoothed one-hot target vectors.

    Args:
        y_true: torch.Tensor of shape (N,) with values in [0, K-1]
        num_classes: int, total number of classes (K)
        epsilon: float in [0, 1]

    Returns:
        torch.Tensor of shape (N, K) with smoothed probabilities.
    """
    # 1. Standardize shape and type constraints
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true, dtype=torch.long)
    else:
        y_true = y_true.long()
        
    N = y_true.size(0)
    
    # 2. Generate a standard uniform background noise matrix component: epsilon / K
    smoothed = torch.full((N, num_classes), fill_value=(epsilon / num_classes), dtype=torch.float32)
    
    # 3. Layer in the remaining high-confidence mass for the ground-truth locations
    # target = (1 - epsilon) + (epsilon / K)
    confidence_mass = 1.0 - epsilon
    smoothed.scatter_(dim=1, index=y_true.unsqueeze(1), value=confidence_mass + (epsilon / num_classes))
    
    return smoothed


def label_smoothing_cross_entropy(logits: torch.Tensor, y_true: torch.Tensor, num_classes: int,
                                  epsilon: float = 0.1, round_decimals: int = None) -> float:
    """
    Compute mean cross-entropy between logits and smoothed targets using stable log-softmax.

    Args:
        logits: torch.Tensor of shape (N, K), model output scores.
        y_true: torch.Tensor of shape (N,), integer class indices.
        num_classes: int, number of classes (K).
        epsilon: float in [0, 1].
        round_decimals: int | None, round the loss to this many decimals if given.

    Returns:
        float: Mean cross-entropy loss.
    """
    # Standardize input primitives into floating-point tensors
    if not isinstance(logits, torch.Tensor):
        logits = torch.tensor(logits, dtype=torch.float32)
    else:
        logits = logits.float()
        
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true, dtype=torch.long)
        
    # 1. Compute numerically stable log probabilities across channels
    log_probs = F.log_softmax(logits, dim=-1)
    
    # 2. Extract our target probability distributions matrix
    targets = smooth_labels(y_true, num_classes, epsilon).to(logits.device)
    
    # 3. Calculate Cross-Entropy: - \sum (target * log_prob) element-wise
    # Then take the mean across the batch dimension
    loss_tensor = -torch.sum(targets * log_probs, dim=-1)
    mean_loss = torch.mean(loss_tensor).item()
    
    # 4. Handle rounding formatting parameter constraints if specified
    if round_decimals is not None:
        mean_loss = round(mean_loss, round_decimals)
        
    return mean_loss