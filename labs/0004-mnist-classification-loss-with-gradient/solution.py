import numpy as np

def loss_function(preds: np.ndarray, target: np.ndarray, reduction: str = "mean", **kwargs):
    """
    preds:     [N, C] softmax probabilities (rows sum to 1)
    target:    [N]    class indices (int64)
    reduction: how to aggregate per-sample losses:
               "mean" → average over batch (gradient scaled by 1/N)
               "sum"  → sum over batch (gradient unscaled)
               "none" → return per-sample loss vector (no aggregation)
    **kwargs:  absorbs any extra arguments from the training harness

    Returns: (loss, grad) where grad has the same shape as preds
    """
    N, C = preds.shape
    
    # Create one-hot encoding of targets
    one_hot = np.zeros_like(preds)
    one_hot[np.arange(N), target] = 1.0
    
    # Compute cross-entropy loss per sample: -log(pred[target])
    # Add small epsilon to avoid log(0)
    eps = 1e-15
    preds_clipped = np.clip(preds, eps, 1.0 - eps)
    per_sample_loss = -np.log(preds_clipped[np.arange(N), target])
    
    # Aggregate loss
    if reduction == "mean":
        loss = np.mean(per_sample_loss)
    elif reduction == "sum":
        loss = np.sum(per_sample_loss)
    elif reduction == "none":
        loss = per_sample_loss
    else:
        raise ValueError(f"Unknown reduction: {reduction}")
    
    # Compute gradient: dL/dpreds = - one_hot / preds (per sample)
    # For cross-entropy with softmax inputs
    grad = -one_hot / preds_clipped
    
    # Scale gradient based on reduction
    if reduction == "mean":
        grad = grad / N
    # For "sum" and "none", keep as is
    
    return loss, grad