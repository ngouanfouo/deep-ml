import torch

def train_logreg(X: torch.Tensor, y: torch.Tensor, learning_rate: float, iterations: int) -> tuple[list[float], list[float]]:
    """
    Train logistic regression using gradient descent with BCE loss.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features) as Tensor
        y: Binary labels of shape (n_samples,) as Tensor
        learning_rate: Step size for gradient descent
        iterations: Number of training iterations
    
    Returns:
        Tuple of (coefficients, losses) where:
        - coefficients: List of learned weights (bias first, then feature weights), rounded to 4 decimals
        - losses: List of BCE loss values at each iteration, rounded to 4 decimals
    
    Notes:
        - Initialize all coefficients to zero
        - Add bias column as FIRST column of X
        - Use sum-based BCE loss: -sum(y*log(p) + (1-y)*log(1-p))
    """
    # Add bias column of ones as the first column
    n_samples, n_features = X.shape
    X_with_bias = torch.cat([torch.ones((n_samples, 1)), X], dim=1)  # shape (n_samples, n_features + 1)
    
    # Initialize weights to zero (n_features + 1)
    weights = torch.zeros(n_features + 1, dtype=torch.float32)
    
    # Store loss values
    losses = []
    
    # Gradient descent
    for _ in range(iterations):
        # Compute logits (linear combination)
        logits = X_with_bias @ weights  # shape (n_samples,)
        
        # Apply sigmoid activation to get probabilities
        # Sigmoid: 1 / (1 + exp(-logits))
        probs = 1 / (1 + torch.exp(-logits))
        
        # Compute sum-based Binary Cross Entropy loss
        # BCE = -sum(y * log(p) + (1-y) * log(1-p))
        # Add small epsilon to avoid log(0)
        eps = 1e-15
        loss = -torch.sum(y * torch.log(probs + eps) + (1 - y) * torch.log(1 - probs + eps))
        losses.append(loss.item())
        
        # Compute gradient: dL/dw = sum((p - y) * X_i)
        gradient = X_with_bias.T @ (probs - y)  # shape (n_features + 1,)
        
        # Update weights
        weights = weights - learning_rate * gradient
    
    # Convert to list and round to 4 decimal places
    coefficients = [round(w.item(), 4) for w in weights]
    losses_rounded = [round(l, 4) for l in losses]
    
    return coefficients, losses_rounded