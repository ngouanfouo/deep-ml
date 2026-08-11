import torch

def diffusion_loss(x_0: torch.Tensor, t: int, beta_start: float, beta_end: float, num_timesteps: int, noise: torch.Tensor, predicted_noise: torch.Tensor) -> float:
    """
    Compute the reconstruction loss for diffusion model training.
    
    Args:
        x_0: Original input data (torch.Tensor)
        t: Timestep (1-indexed, from 1 to num_timesteps)
        beta_start: Starting value of linear beta schedule
        beta_end: Ending value of linear beta schedule
        num_timesteps: Total number of diffusion timesteps
        noise: True noise tensor (same shape as x_0)
        predicted_noise: Model's predicted noise (same shape as x_0)
    
    Returns:
        Mean squared error loss (float)
    """
    # Create linear beta schedule
    betas = torch.linspace(beta_start, beta_end, num_timesteps)
    
    # Compute alphas
    alphas = 1 - betas
    
    # Compute cumulative product of alphas (alpha_bar)
    alpha_bar = torch.cumprod(alphas, dim=0)
    
    # Get alpha_bar for timestep t (adjust for 0-indexing)
    alpha_bar_t = alpha_bar[t - 1]
    
    # Compute noisy sample x_t from x_0 using the forward process
    # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
    sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
    sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)
    x_t = sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise
    
    # Reconstruct x_0 from x_t using the predicted noise
    # x_0_reconstructed = (x_t - sqrt(1 - alpha_bar_t) * predicted_noise) / sqrt(alpha_bar_t)
    x_0_reconstructed = (x_t - sqrt_one_minus_alpha_bar_t * predicted_noise) / sqrt_alpha_bar_t
    
    # Compute mean squared error between original and reconstructed x_0
    loss = torch.mean((x_0 - x_0_reconstructed) ** 2)
    
    # Convert to Python float
    return loss.item()