import torch

def square_relu(x: torch.Tensor) -> dict:
    """
    Apply the Square ReLU activation function and compute its derivative.
    
    Args:
        x: Input torch.Tensor of any shape
    
    Returns:
        Dictionary with 'output' and 'derivative' as torch.Tensors
    """
    # Apply standard ReLU to the input
    relu_x = torch.relu(x)
    
    # Square ReLU output: (ReLU(x))^2
    output = relu_x ** 2
    
    # Derivative of Square ReLU: 2 * ReLU(x)
    derivative = 2.0 * relu_x
    
    return {
        'output': torch.round(output, decimals=4),
        'derivative': torch.round(derivative, decimals=4)
    }