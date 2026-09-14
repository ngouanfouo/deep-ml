import torch

def ddpm_reverse_step(x_t: torch.Tensor, predicted_noise: torch.Tensor, t: int,
                     betas: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
    """
    Perform a single reverse (denoising) step of the DDPM sampling process.
    
    Args:
        x_t: Noisy sample at timestep t, shape (D,)
        predicted_noise: Model's noise prediction, shape (D,)
        t: Current timestep (1-indexed)
        betas: Noise schedule tensor, shape (T,)
        noise: Optional noise tensor for stochastic component, shape (D,)
    
    Returns:
        Denoised sample x_{t-1}, shape (D,)
    """
    idx = t - 1
    beta_t = betas[idx]
    alpha_t = 1.0 - beta_t
    
    # Compute alphas and alpha_bars up to timestep t
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    alpha_bar_t = alpha_bars[idx]
    
    # 1. Compute the DDPM posterior mean parameterization
    coef = beta_t / torch.sqrt(1.0 - alpha_bar_t)
    mean = (1.0 / torch.sqrt(alpha_t)) * (x_t - coef * predicted_noise)
    
    # 2. At the final step (t = 1), no stochastic noise is added
    if t == 1:
        return mean
        
    # 3. For t > 1, add stochastic noise scaled by sqrt(beta_t)
    sigma = torch.sqrt(beta_t)
    
    if noise is None:
        noise = torch.randn_like(x_t)
        
    return mean + sigma * noise