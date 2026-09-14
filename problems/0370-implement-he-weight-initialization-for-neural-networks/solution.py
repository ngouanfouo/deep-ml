import numpy as np
import torch
import torch.nn as nn

def he_initialize(layer_dims: list, method: str = 'normal', seed: int = 42) -> list:
    """
    Initialize weight matrices for a neural network using He (Kaiming) initialization.
    
    Args:
        layer_dims: List of integers representing neurons per layer
        method: 'normal' or 'uniform' sampling distribution
        seed: Random seed for reproducibility
    
    Returns:
        List of torch.Tensors, one weight matrix per adjacent layer pair
    """
    # Set the random seed once at the beginning using numpy as requested
    np.random.seed(seed)
    
    weights = []
    
    # Iterate through adjacent layer pairs to create weight matrices
    for i in range(len(layer_dims) - 1):
        fan_in = layer_dims[i]
        fan_out = layer_dims[i + 1]
        
        if method == 'normal':
            # Standard deviation for He normal: sqrt(2 / fan_in)
            std = np.sqrt(2.0 / fan_in)
            w = np.random.normal(0.0, std, size=(fan_in, fan_out))
        elif method == 'uniform':
            # Bound limit for He uniform: sqrt(6 / fan_in)
            bound = np.sqrt(6.0 / fan_in)
            w = np.random.uniform(-bound, bound, size=(fan_in, fan_out))
        else:
            raise ValueError("method must be 'normal' or 'uniform'")
            
        # Convert numpy array to PyTorch tensor to support torch operations
        weights.append(torch.tensor(w, dtype=torch.float32))
        
    return weights