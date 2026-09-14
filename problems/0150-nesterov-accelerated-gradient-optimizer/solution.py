import torch

def nag_optimizer(parameter: torch.Tensor, grad_fn, velocity: torch.Tensor, learning_rate: float = 0.01, momentum: float = 0.9) -> tuple:
    """
    Update parameters using the Nesterov Accelerated Gradient optimizer.
    Uses a "look-ahead" approach to improve convergence by applying momentum before computing the gradient.
    Args:
        parameter: Current parameter value (torch.Tensor)
        grad_fn: Function that computes the gradient at a given position
        velocity: Current velocity (momentum term) (torch.Tensor)
        learning_rate: Learning rate (default=0.01)
        momentum: Momentum coefficient (default=0.9)
    Returns:
        tuple: (updated_parameter, updated_velocity) as torch.Tensors
    """
    # 1. Compute the look-ahead parameter position
    parameter_lookahead = parameter - momentum * velocity
    
    # 2. Compute the gradient at the look-ahead position
    grad = grad_fn(parameter_lookahead)
    
    # 3. Update the velocity (momentum term)
    updated_velocity = momentum * velocity + learning_rate * grad
    
    # 4. Update the parameters using the new velocity
    updated_parameter = parameter - updated_velocity
    
    return torch.round(updated_parameter, decimals=5), torch.round(updated_velocity, decimals=5)