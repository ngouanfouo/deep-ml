import torch

def predict_logistic(X: torch.Tensor, weights: torch.Tensor, bias: float) -> torch.Tensor:
    """
    Implements binary classification prediction using Logistic Regression.

    Args:
        X: Input feature matrix (shape: N x D)
        weights: Model weights (shape: D)
        bias: Model bias

    Returns:
        Binary predictions (0 or 1)
    """
    # Compute linear combination
    z = torch.matmul(X, weights) + bias
    
    # Apply sigmoid function
    probabilities = torch.sigmoid(z)
    
    # Apply threshold of 0.5 (>= for tie case)
    predictions = (probabilities >= 0.5).int()
    
    return predictions