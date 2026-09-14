import torch
import torch.nn.functional as F

def noise_prediction_loss(
    x_0: torch.Tensor,
    alpha_bar: torch.Tensor,
    t: torch.Tensor,
    epsilon: torch.Tensor,
    epsilon_pred: torch.Tensor
) -> tuple:
    """
    Compute the noisy samples and noise prediction MSE loss for diffusion model training.

    Args:
        x_0: Clean data samples, shape (B, D)
        alpha_bar: Cumulative noise schedule, shape (T,)
        t: Timestep indices for each sample, shape (B,)
        epsilon: True Gaussian noise, shape (B, D)
        epsilon_pred: Predicted noise from model, shape (B, D)

    Returns:
        tuple: (x_t, loss) where x_t is a torch.Tensor of shape (B, D)
               and loss is a scalar torch.Tensor
    """
    # 1. Look up alpha_bar for each sample's timestep
    alpha_bar_t = alpha_bar[t]  # Shape: (B,)
    
    # Reshape coefficients for broadcasting across feature dimensions (B, D)
    sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t).unsqueeze(-1)
    sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t).unsqueeze(-1)
    
    # 2. Construct the noisy version of each sample using the closed-form reparameterization
    x_t = sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * epsilon
    
    # 3. Compute the mean squared error loss between actual noise and predicted noise
    loss = F.mse_loss(epsilon_pred, epsilon)
    
    return x_t, loss