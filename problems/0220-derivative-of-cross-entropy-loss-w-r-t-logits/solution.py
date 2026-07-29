import numpy as np

def cross_entropy_derivative(logits: list[float], target: int) -> list[float]:
    """
    Compute the derivative of cross-entropy loss with respect to logits.
    
    Args:
        logits: Raw model outputs (before softmax)
        target: Index of the true class (0-indexed)
        
    Returns:
        Gradient vector where gradient[i] = dL/d(logits[i])
    """
    # Convert to numpy array
    logits = np.array(logits, dtype=np.float64)
    
    # Compute softmax probabilities
    # For numerical stability, subtract max
    logits_shifted = logits - np.max(logits)
    exp_logits = np.exp(logits_shifted)
    probabilities = exp_logits / np.sum(exp_logits)
    
    # Create one-hot encoding for target
    n = len(logits)
    one_hot = np.zeros(n, dtype=np.float64)
    one_hot[target] = 1.0
    
    # Gradient of cross-entropy loss with respect to logits
    # ∂L/∂z_i = p_i - y_i
    gradient = probabilities - one_hot
    
    return gradient.tolist()