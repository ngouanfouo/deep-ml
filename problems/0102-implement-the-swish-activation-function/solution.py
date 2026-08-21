import torch

def swish(x: torch.Tensor) -> torch.Tensor:
    """
    Implements the Swish activation function.

    Args:
        x: Input tensor

    Returns:
        The Swish activation value as a tensor
    """
    return x * torch.sigmoid(x)