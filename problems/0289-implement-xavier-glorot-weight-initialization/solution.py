import torch
import torch.nn as nn

def xavier_initialization(shape: tuple, mode: str = 'uniform', seed: int = None) -> torch.Tensor:
    """
    Implement Xavier/Glorot weight initialization using PyTorch.
    
    Args:
        shape: Tuple of (fan_in, fan_out) representing weight matrix dimensions
        mode: 'uniform' or 'normal' initialization
        seed: Random seed for reproducibility (optional)
    
    Returns:
        Initialized weight matrix as torch.Tensor
    
    Example:
        >>> xavier_initialization((3, 3), 'uniform', seed=42)
        tensor([[-0.2509,  0.9014,  0.4640],
                [ 0.1973, -0.6880, -0.6880],
                [-0.8838,  0.7324,  0.2022]])
    """
    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)
    
    # Unpack shape
    fan_in, fan_out = shape
    
    # Calculate the scale factor
    # For Xavier/Glorot initialization:
    # Uniform: limit = sqrt(6 / (fan_in + fan_out))
    # Normal: std = sqrt(2 / (fan_in + fan_out))
    # Note: Some implementations use sqrt(2/(fan_in + fan_out)) for normal,
    # and sqrt(6/(fan_in + fan_out)) for uniform
    
    if mode == 'uniform':
        # Xavier uniform initialization
        # Weights are drawn from Uniform(-limit, limit)
        limit = torch.sqrt(torch.tensor(6.0 / (fan_in + fan_out)))
        weights = torch.empty(shape).uniform_(-limit, limit)
        
    elif mode == 'normal':
        # Xavier normal initialization
        # Weights are drawn from Normal(0, std)
        std = torch.sqrt(torch.tensor(2.0 / (fan_in + fan_out)))
        weights = torch.empty(shape).normal_(0, std)
        
    else:
        raise ValueError(f"mode must be 'uniform' or 'normal', got {mode}")
    
    return weights