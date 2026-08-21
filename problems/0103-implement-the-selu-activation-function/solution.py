import torch

def selu(x: torch.Tensor) -> torch.Tensor:
    """
    Implements the SELU (Scaled Exponential Linear Unit) activation function.

    Args:
        x: Input tensor

    Returns:
        SELU activation tensor
    """
    alpha = 1.6732632423543772
    scale = 1.0507009873554804
    
    # Apply SELU: scale * (x if x > 0 else alpha * (exp(x) - 1))
    return scale * torch.where(x > 0, x, alpha * (torch.exp(x) - 1))