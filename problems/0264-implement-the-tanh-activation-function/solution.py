import torch

def tanh(x: torch.Tensor) -> torch.Tensor:
    """
    Implements the Tanh (hyperbolic tangent) activation function.

    Args:
        x (torch.Tensor): Input tensor

    Returns:
        torch.Tensor: The tanh of the input, rounded to 4 decimal places
    """
    # Compute tanh and round to 4 decimal places
    return torch.round(torch.tanh(x), decimals=4)