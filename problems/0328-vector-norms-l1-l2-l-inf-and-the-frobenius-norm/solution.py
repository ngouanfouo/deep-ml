import torch

def compute_norm(arr: torch.Tensor, norm_type: str) -> float:
    """
    Compute the specified norm of the input tensor.
    
    Args:
        arr: Input tensor (1D or 2D)
        norm_type: Type of norm ('l1', 'l2', or 'frobenius')
    
    Returns:
        The computed norm as a float
    """
    if not isinstance(arr, torch.Tensor):
        arr = torch.tensor(arr, dtype=torch.float32)
    else:
        arr = arr.float()
        
    norm_type = norm_type.lower()
    
    if norm_type == 'l1':
        # L1 Norm: Sum of absolute values
        result = torch.sum(torch.abs(arr))
    elif norm_type == 'l2':
        # L2 Norm: Square root of sum of squared values
        result = torch.sqrt(torch.sum(arr ** 2))
    elif norm_type == 'frobenius':
        # Frobenius Norm: Square root of sum of absolute squares of all elements
        result = torch.sqrt(torch.sum(arr ** 2))
    else:
        raise ValueError(f"Unsupported norm_type: {norm_type}")
        
    return float(result.item())