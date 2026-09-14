import torch
from typing import Tuple

def momentum_optimizer(parameter: torch.Tensor, grad: torch.Tensor, velocity: torch.Tensor,
                       learning_rate: float = 0.01, momentum: float = 0.9) -> Tuple[torch.Tensor, torch.Tensor]:
    if not torch.is_tensor(parameter):
        parameter = torch.as_tensor(parameter)
    if not torch.is_tensor(grad):
        grad = torch.as_tensor(grad, dtype=parameter.dtype, device=parameter.device)
    if not torch.is_tensor(velocity):
        velocity = torch.as_tensor(velocity, dtype=parameter.dtype, device=parameter.device)

    updated_velocity = momentum * velocity + learning_rate * grad
    updated_parameter = parameter - updated_velocity
    return updated_parameter, updated_velocity