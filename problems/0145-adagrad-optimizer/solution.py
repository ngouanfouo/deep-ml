import torch

def adagrad_optimizer(parameter: torch.Tensor, grad: torch.Tensor, G: torch.Tensor,
                      learning_rate: float = 0.01, epsilon: float = 1e-8) -> tuple:
    """
    Update parameters using the Adagrad optimizer.
    Adapts the learning rate for each parameter based on the historical gradients.
    """
    # --- Input validation ---
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    
    # Coerce inputs to float tensors (handles scalars, lists, numpy arrays)
    if not isinstance(parameter, torch.Tensor):
        parameter = torch.tensor(parameter, dtype=torch.float32)
    if not isinstance(grad, torch.Tensor):
        grad = torch.tensor(grad, dtype=torch.float32)
    if not isinstance(G, torch.Tensor):
        G = torch.tensor(G, dtype=torch.float32)
    
    # Ensure floating-point dtype (integers would break sqrt and division)
    parameter = parameter.to(torch.float32)
    grad = grad.to(torch.float32)
    G = G.to(torch.float32)
    
    # Check shape compatibility
    if parameter.shape != grad.shape:
        raise ValueError(f"parameter shape {parameter.shape} != grad shape {grad.shape}")
    if parameter.shape != G.shape:
        raise ValueError(f"parameter shape {parameter.shape} != G shape {G.shape}")
    
    # --- Adagrad update ---
    # 1. Accumulate squared gradients
    G_new = G + grad ** 2
    
    # 2. Update parameter with per-element adaptive learning rate
    parameter_new = parameter - learning_rate * grad / (torch.sqrt(G_new) + epsilon)
    
    return torch.round(parameter_new, decimals=5), torch.round(G_new, decimals=5)