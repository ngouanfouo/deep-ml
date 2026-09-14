import torch

def ddim_sample_step(x_t: torch.Tensor, noise_pred: torch.Tensor, alpha_bar_t: float, alpha_bar_t_prev: float) -> tuple:
    """
    Perform one deterministic DDIM sampling step.

    Args:
        x_t: Noisy sample at timestep t, shape (D,)
        noise_pred: Predicted noise from the model, shape (D,)
        alpha_bar_t: Cumulative alpha at timestep t
        alpha_bar_t_prev: Cumulative alpha at timestep t-1

    Returns:
        Tuple of (pred_x0, x_t_prev), each a torch.Tensor rounded to 4 decimals
    """
    device = x_t.device
    dtype = x_t.dtype
    
    # Cast schedule scalars to tensors with matching device and dtype
    sqrt_alpha_bar_t = torch.sqrt(torch.tensor(alpha_bar_t, dtype=dtype, device=device))
    sqrt_one_minus_alpha_bar_t = torch.sqrt(torch.tensor(1.0 - alpha_bar_t, dtype=dtype, device=device))
    
    # 1. Predict clean sample x_0
    pred_x0 = (x_t - sqrt_one_minus_alpha_bar_t * noise_pred) / sqrt_alpha_bar_t
    
    # 2. Compute deterministic x_{t-1} using DDIM formula (eta = 0)
    sqrt_alpha_bar_prev = torch.sqrt(torch.tensor(alpha_bar_t_prev, dtype=dtype, device=device))
    sqrt_one_minus_alpha_bar_prev = torch.sqrt(torch.tensor(1.0 - alpha_bar_t_prev, dtype=dtype, device=device))
    
    x_t_prev = sqrt_alpha_bar_prev * pred_x0 + sqrt_one_minus_alpha_bar_prev * noise_pred
    
    # 3. Round both tensors to 4 decimal places
    pred_x0 = torch.round(pred_x0, decimals=4)
    x_t_prev = torch.round(x_t_prev, decimals=4)
    
    return pred_x0, x_t_prev