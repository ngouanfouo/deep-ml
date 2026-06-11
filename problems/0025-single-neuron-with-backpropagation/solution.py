import torch

def train_neuron(features: torch.Tensor, labels: torch.Tensor, initial_weights: torch.Tensor, initial_bias: float, learning_rate: float, epochs: int) -> tuple[list[float], float, list[float]]:
    """
    Simulates a single neuron with sigmoid activation and trains it using
    backpropagation with MSE loss via full-batch gradient descent.

    Args:
        features: Input feature tensor of shape (n_samples, n_features)
        labels: Binary label tensor of shape (n_samples,)
        initial_weights: Initial weight tensor of shape (n_features,)
        initial_bias: Initial bias scalar
        learning_rate: Learning rate for SGD
        epochs: Number of training epochs

    Returns:
        Tuple of (updated_weights, updated_bias, mse_values) all rounded to 4 decimal places
    """
    # Force float64 tensors to ensure numerical precision matching standard float math
    w = initial_weights.clone().to(torch.float64)
    b = torch.tensor(initial_bias, dtype=torch.float64)
    X = features.to(torch.float64)
    y = labels.to(torch.float64)
    
    n_samples = X.shape[0]
    mse_values = []
    
    for epoch in range(epochs):
        # 1. Forward pass: compute linear combination and apply sigmoid activation
        # z shape: (n_samples,)
        z = torch.mv(X, w) + b
        a = 1.0 / (1.0 + torch.exp(-z))
        
        # 2. Compute Mean Squared Error (MSE) before the update
        mse = torch.mean((a - y) ** 2)
        mse_values.append(round(mse.item(), 4))
        
        # 3. Backward pass: compute analytical gradients averaged over the batch
        # error_signal (delta) shape: (n_samples,)
        error_signal = (2.0 / n_samples) * (a - y) * a * (1.0 - a)
        
        # Gradients
        dw = torch.mv(X.T, error_signal)
        db = torch.sum(error_signal)
        
        # 4. Gradient descent parameter updates
        w -= learning_rate * dw
        b -= learning_rate * db

    # Format and round outputs to 4 decimal places
    updated_weights = [round(val, 4) for val in w.tolist()]
    updated_bias = round(b.item(), 4)
    
    return updated_weights, updated_bias, mse_values

