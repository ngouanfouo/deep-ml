import torch
from typing import Callable, Tuple

def numerical_gradient_check(f: Callable, x: torch.Tensor, analytical_grad: torch.Tensor, epsilon: float = 1e-7) -> Tuple[torch.Tensor, float]:
    """
    Perform numerical gradient checking using centered finite differences.

    Args:
        f: A function that takes a torch.Tensor and returns a scalar
        x: torch.Tensor, the point at which to check gradient
        analytical_grad: torch.Tensor, the analytically computed gradient
        epsilon: float, small value for finite difference approximation

    Returns:
        tuple: (numerical_grad, relative_error)
    """
    # Ensure we're working with float tensors
    if x.dtype != torch.float64 and x.dtype != torch.float32:
        x = x.float()
    if analytical_grad.dtype != torch.float64 and analytical_grad.dtype != torch.float32:
        analytical_grad = analytical_grad.float()
    
    # Create a copy without gradient tracking
    x_original = x.detach().clone()
    analytical_grad = analytical_grad.detach().clone()
    
    # Initialize numerical gradient tensor
    numerical_grad = torch.zeros_like(x_original)
    
    # Iterate over each dimension
    for i in range(x_original.numel()):
        # Create perturbation vectors
        x_plus = x_original.clone()
        x_minus = x_original.clone()
        
        # Perturb the i-th dimension
        x_plus_flat = x_plus.view(-1)
        x_minus_flat = x_minus.view(-1)
        x_plus_flat[i] += epsilon
        x_minus_flat[i] -= epsilon
        
        # Evaluate function at perturbed points
        f_plus = f(x_plus)
        f_minus = f(x_minus)
        
        # Extract scalar values
        if isinstance(f_plus, torch.Tensor):
            f_plus_val = f_plus.item()
        else:
            f_plus_val = float(f_plus)
            
        if isinstance(f_minus, torch.Tensor):
            f_minus_val = f_minus.item()
        else:
            f_minus_val = float(f_minus)
        
        # Centered difference approximation
        numerical_grad_flat = numerical_grad.view(-1)
        numerical_grad_flat[i] = (f_plus_val - f_minus_val) / (2 * epsilon)
    
    # Compute relative error
    diff = analytical_grad - numerical_grad
    diff_norm = torch.norm(diff).item()
    
    analytical_norm = torch.norm(analytical_grad).item()
    numerical_norm = torch.norm(numerical_grad).item()
    
    # Handle edge case where both gradients are zero
    if analytical_norm == 0 and numerical_norm == 0:
        relative_error = 0.0
    else:
        # Add small epsilon to denominator to avoid division by zero
        denominator = analytical_norm + numerical_norm + 1e-12
        relative_error = diff_norm / denominator
    
    return numerical_grad, relative_error