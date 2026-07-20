import torch
import torch.nn.functional as F

def overlapping_max_pool2d(x: torch.Tensor, kernel_size: int = 3, stride: int = 2) -> torch.Tensor:
    """
    Apply overlapping max pooling using PyTorch.
    Must match the ceil mode behavior described in the problem.
    
    Args:
        x: Input tensor of shape (N, C, H, W)
        kernel_size: Size of the pooling window
        stride: Stride of the pooling window (stride < kernel_size)
        
    Returns:
        torch.Tensor: Pooled tensor with dimensions determined by ceil mode.
    """
    # Enforce float type for numerical consistency if necessary
    if not x.is_floating_point():
        x = x.to(torch.float32)
        
    # PyTorch's F.max_pool2d supports ceil_mode directly out-of-the-box.
    # When ceil_mode=True, it matches the exact edge-handling and boundary
    # dimension properties described in your specification.
    out = F.max_pool2d(
        x, 
        kernel_size=kernel_size, 
        stride=stride, 
        padding=0, 
        ceil_mode=True
    )
    
    return out