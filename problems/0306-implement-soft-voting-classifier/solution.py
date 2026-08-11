import torch

def soft_voting_classifier(probabilities: torch.Tensor, weights: list = None) -> torch.Tensor:
    """
    Implement soft voting for ensemble classification.
    
    Args:
        probabilities: 3D tensor of shape (n_classifiers, n_samples, n_classes)
        weights: Optional list of weights for each classifier
    
    Returns:
        torch.Tensor of predicted class labels for each sample
    """
    # Input validation
    if probabilities.numel() == 0:
        return torch.tensor([], dtype=torch.long)
    
    n_classifiers, n_samples, n_classes = probabilities.shape
    
    # Handle weights
    if weights is None:
        # Uniform weights
        weights_tensor = torch.ones(n_classifiers) / n_classifiers
    else:
        # Convert weights to tensor and normalize
        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        weights_tensor = weights_tensor / weights_tensor.sum()
    
    # Ensure weights tensor has correct shape for broadcasting
    # Shape: (n_classifiers, 1, 1)
    weights_tensor = weights_tensor.view(-1, 1, 1)
    
    # Compute weighted average of probabilities
    # weighted_probs = sum(weight_i * prob_i) for all i
    weighted_probs = (probabilities * weights_tensor).sum(dim=0)
    
    # Get predicted class by taking argmax along class dimension
    predictions = weighted_probs.argmax(dim=1)
    
    return predictions