import torch
import math
from typing import Dict, Union, List

def xavier_init(fan_in: int, fan_out: int, mode: str = 'uniform', seed: int = 42) -> Dict[str, Union[List[List[float]], List[int], float]]:
    """
    Perform Xavier/Glorot weight initialization using PyTorch built-ins.

    Args:
        fan_in (int): Number of input units.
        fan_out (int): Number of output units.
        mode (str): 'uniform' or 'normal'.
        seed (int): Random seed for reproducibility.

    Returns:
        dict: Contains 'weights' (nested list), 'shape' (list), and 'param' (float).
    """
    if mode not in ['uniform', 'normal']:
        raise ValueError("mode must be 'uniform' or 'normal'")

    # Set random seed for reproducibility
    torch.manual_seed(seed)

    if mode == 'uniform':
        # Limit for uniform distribution: sqrt(6 / (fan_in + fan_out))
        param = math.sqrt(6.0 / (fan_in + fan_out))
        weights = torch.empty(fan_in, fan_out).uniform_(-param, param)
    else:
        # Standard deviation for normal distribution: sqrt(2 / (fan_in + fan_out))
        param = math.sqrt(2.0 / (fan_in + fan_out))
        weights = torch.empty(fan_in, fan_out).normal_(0.0, param)

    # Convert weights tensor to a nested list and round elements to 4 decimal places
    weights_list = [[round(val.item(), 4) for val in row] for row in weights]

    return {
        'weights': weights_list,
        'shape': [fan_in, fan_out],
        'param': round(param, 4)
    }