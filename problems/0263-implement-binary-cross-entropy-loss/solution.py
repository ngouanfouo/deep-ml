import math

def binary_cross_entropy(y_true: list[float], y_pred: list[float], epsilon: float = 1e-15) -> float:
    """
    Compute binary cross-entropy loss.
    
    Args:
        y_true: True binary labels (0 or 1)
        y_pred: Predicted probabilities (between 0 and 1)
        epsilon: Small value for numerical stability
    
    Returns:
        Mean binary cross-entropy loss
    """
    # Clip predicted probabilities to avoid log(0) or log(1)
    y_pred_clipped = [max(min(p, 1.0 - epsilon), epsilon) for p in y_pred]
    
    # Compute the binary cross-entropy loss for each sample
    losses = [
        -(y * math.log(p) + (1.0 - y) * math.log(1.0 - p))
        for y, p in zip(y_true, y_pred_clipped)
    ]
    
    # Return the mean loss across all samples
    return sum(losses) / len(losses) if losses else 0.0