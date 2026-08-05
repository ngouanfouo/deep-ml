import torch

def focal_loss(y_true: torch.Tensor, y_pred: torch.Tensor, gamma: float = 2.0, alpha: torch.Tensor = None) -> float:
    """
    Compute Focal Loss for multi-class classification.
    
    Args:
        y_true: Ground truth labels as class indices (1D tensor)
        y_pred: Predicted probabilities (2D tensor, shape: [n_samples, n_classes])
        gamma: Focusing parameter (default: 2.0)
        alpha: Class weights (optional, 1D tensor of length n_classes)
    
    Returns:
        float: Average focal loss
    """
    # Clip predictions to avoid numerical issues with log(0)
    epsilon = 1e-8
    y_pred = torch.clamp(y_pred, epsilon, 1.0 - epsilon)
    
    # Get the predicted probability for the true class
    # y_true: (n_samples,), y_pred: (n_samples, n_classes)
    # Use gather to index into y_pred along dim=1 with y_true as indices
    pt = y_pred.gather(dim=1, index=y_true.unsqueeze(1)).squeeze(1)
    
    # Compute cross-entropy: -log(pt)
    ce_loss = -torch.log(pt)
    
    # Compute focal weight: (1 - pt)^gamma
    focal_weight = (1.0 - pt) ** gamma
    
    # Compute focal loss
    if alpha is not None:
        # Apply class weights
        # Get alpha for each sample's true class
        alpha_t = alpha.gather(dim=0, index=y_true)
        focal_loss_per_sample = alpha_t * focal_weight * ce_loss
    else:
        focal_loss_per_sample = focal_weight * ce_loss
    
    # Return average focal loss
    return focal_loss_per_sample.mean().item()