import torch

def neural_memory_update(
    M: torch.Tensor,
    S: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    theta: float = 0.1,
    eta: float = 0.9,
    alpha: float = 0.01
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Update neural memory using surprise-based learning with momentum and forgetting.
    
    Args:
        M: Current memory state matrix of shape (d, d)
        S: Current momentum/surprise accumulator of shape (d, d)
        k: Key vector of shape (d,)
        v: Value vector of shape (d,)
        theta: Learning rate for momentary surprise (default: 0.1)
        eta: Momentum decay factor for past surprise (default: 0.9)
        alpha: Forget gate - fraction of old memory to forget (default: 0.01)
    
    Returns:
        Tuple of (updated_M, updated_S) where:
        - updated_M: New memory state after update
        - updated_S: New momentum state after update
    """
    # Ensure inputs are torch tensors
    if not isinstance(M, torch.Tensor):
        M = torch.tensor(M, dtype=torch.float32)
    if not isinstance(S, torch.Tensor):
        S = torch.tensor(S, dtype=torch.float32)
    if not isinstance(k, torch.Tensor):
        k = torch.tensor(k, dtype=torch.float32)
    if not isinstance(v, torch.Tensor):
        v = torch.tensor(v, dtype=torch.float32)
    
    # Ensure proper shapes
    if M.dim() == 0 or M.dim() > 2:
        raise ValueError(f"M should be a 2D matrix, got shape {M.shape}")
    
    if S.shape != M.shape:
        raise ValueError(f"S shape {S.shape} must match M shape {M.shape}")
    
    if k.dim() == 0 or k.dim() > 1:
        raise ValueError(f"k should be a 1D vector, got shape {k.shape}")
    
    if v.shape != k.shape:
        raise ValueError(f"v shape {v.shape} must match k shape {k.shape}")
    
    # Compute prediction: M @ k
    # M: (d, d), k: (d,) -> prediction: (d,)
    prediction = M @ k
    
    # Compute prediction error: (M @ k - v)
    # prediction: (d,), v: (d,) -> error: (d,)
    error = prediction - v
    
    # Compute momentary surprise (gradient of loss ||M @ k - v||^2 with respect to M)
    # Gradient = (M @ k - v) @ k.T
    # error: (d,), k: (d,) -> gradient: (d, d)
    momentary_surprise = torch.outer(error, k)
    
    # Update momentum: S_new = eta * S - theta * momentary_surprise
    # Note: The formula uses negative sign for gradient descent
    # eta: decay factor for past momentum, theta: learning rate
    updated_S = eta * S - theta * momentary_surprise
    
    # Update memory with forgetting: M_new = (1 - alpha) * M + S_new
    # alpha: forget gate (1-alpha) retains old memory, S_new adds the update
    updated_M = (1 - alpha) * M + updated_S
    
    return updated_M, updated_S