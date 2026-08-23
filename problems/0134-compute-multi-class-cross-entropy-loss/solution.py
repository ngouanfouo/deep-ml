import torch

def compute_cross_entropy_loss(predicted_probs: torch.Tensor, true_labels: torch.Tensor, epsilon: float = 1e-15) -> float:
    """Compute average cross-entropy loss for multi-class classification.
    
    Args:
        predicted_probs: Tensor of predicted probabilities (batch_size, num_classes)
        true_labels: One-hot encoded true labels (batch_size, num_classes)
        epsilon: Small value for numerical stability
    
    Returns:
        Average cross-entropy loss as a float
    """
    # Ensure inputs are tensors
    if not isinstance(predicted_probs, torch.Tensor):
        predicted_probs = torch.tensor(predicted_probs, dtype=torch.float32)
    if not isinstance(true_labels, torch.Tensor):
        true_labels = torch.tensor(true_labels, dtype=torch.float32)
        
    # Clip probabilities for numerical stability to avoid log(0)
    preds_clipped = torch.clamp(predicted_probs, min=epsilon, max=1.0 - epsilon)
    
    # Compute the cross-entropy loss: - sum(y * log(p)) over classes, then mean over the batch
    loss_per_sample = -torch.sum(true_labels * torch.log(preds_clipped), dim=1)
    average_loss = torch.mean(loss_per_sample)
    
    return float(average_loss.item())