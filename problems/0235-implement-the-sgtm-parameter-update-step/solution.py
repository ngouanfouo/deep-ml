import torch

def sgtm_step(
    params: torch.Tensor,
    grad: torch.Tensor,
    forget_mask: torch.Tensor,
    lr: float = 0.1,
    batch_type: str = "forget",
) -> torch.Tensor:
    """Perform one SGTM update step on a 1D parameter tensor.

    See the NumPy version for full details.

    Args:
        params: Current parameters, shape (d,)
        grad: Gradient for this batch, shape (d,)
        forget_mask: Mask for forget parameters, shape (d,)
        lr: Learning rate
        batch_type: One of {'forget', 'retain', 'unlabeled'}

    Returns:
        new_params: Updated parameters as a tensor
    """
    # Convert mask to boolean type for indexing
    forget_mask_bool = forget_mask.bool()
    retain_mask_bool = ~forget_mask_bool
    
    # Initialize update mask with zeros (same shape as params)
    update_mask = torch.zeros_like(params, dtype=torch.bool)
    
    if batch_type == "forget":
        # For 'forget' batches: only update forget parameters
        update_mask = forget_mask_bool
    elif batch_type == "retain":
        # For 'retain' batches: only update retain parameters
        update_mask = retain_mask_bool
    else:  # "unlabeled"
        # For 'unlabeled' batches: update all parameters
        update_mask = torch.ones_like(params, dtype=torch.bool)
    
    # Apply the gradient update: params = params - lr * grad
    # Only apply to parameters where update_mask is True
    # For masked-out parameters, keep the original value
    new_params = params.clone()
    new_params[update_mask] = params[update_mask] - lr * grad[update_mask]
    
    return new_params