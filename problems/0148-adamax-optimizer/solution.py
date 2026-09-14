import torch
from typing import Tuple, Union

def adamax_optimizer(parameter: Union[float, torch.Tensor], 
                     grad: Union[float, torch.Tensor], 
                     m: Union[float, torch.Tensor], 
                     u: Union[float, torch.Tensor], 
                     t: int, 
                     learning_rate: float = 0.002, 
                     beta1: float = 0.9, 
                     beta2: float = 0.999, 
                     epsilon: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Update parameters using the Adamax optimizer.
    Adamax is a variant of Adam based on the infinity norm.
    It uses the maximum of past gradients instead of the L2 norm.
    
    Args:
        parameter: Current parameter value
        grad: Current gradient
        m: First moment estimate
        u: Infinity norm estimate
        t: Current timestep
        learning_rate: Learning rate (default=0.002)
        beta1: First moment decay rate (default=0.9)
        beta2: Infinity norm decay rate (default=0.999)
        epsilon: Small constant for numerical stability (default=1e-8)
        
    Returns:
        tuple: (updated_parameter, updated_m, updated_u)
    """
    # Ensure all inputs are PyTorch tensors for seamless scalar and array handling
    if not isinstance(parameter, torch.Tensor):
        parameter = torch.tensor(parameter, dtype=torch.float32)
    if not isinstance(grad, torch.Tensor):
        grad = torch.tensor(grad, dtype=torch.float32)
    if not isinstance(m, torch.Tensor):
        m = torch.tensor(m, dtype=torch.float32)
    if not isinstance(u, torch.Tensor):
        u = torch.tensor(u, dtype=torch.float32)

    # 1. Update the biased first moment estimate (m)
    m_new = beta1 * m + (1.0 - beta1) * grad

    # 2. Update the exponentially weighted infinity norm (u)
    u_new = torch.max(beta2 * u, torch.abs(grad))

    # 3. Compute bias correction factor for the first moment
    bias_correction = 1.0 - (beta1 ** t)

    # 4. Compute effective step size with bias correction and update parameters
    step_size = learning_rate / bias_correction
    parameter_new = parameter - step_size * (m_new / (u_new + epsilon))

    return parameter_new, m_new, u_new