import torch
import math

def cosine_noise_schedule(T: int, s: float = 0.008, beta_max: float = 0.999) -> dict:
    """
    Compute the cosine noise schedule for a diffusion model.

    Args:
        T: Total number of diffusion timesteps
        s: Small offset to prevent alpha_bar from being too small near t=0
        beta_max: Maximum value for beta clipping

    Returns:
        Dictionary with keys 'betas', 'alphas', 'alpha_bars',
        each containing a torch.Tensor of length T.
    """
    # 1. Define the function f(t) over timesteps t = 0 to T
    steps = torch.arange(T + 1, dtype=torch.float64)
    f_t = torch.cos(((steps / T + s) / (1.0 + s)) * (math.pi / 2.0)) ** 2
    
    # 2. Compute raw cumulative alpha_bars normalized by f(0)
    alpha_bars = f_t / f_t[0]
    
    # 3. Derive per-step betas from consecutive alpha_bar values: beta_t = 1 - (alpha_bar_t / alpha_bar_{t-1})
    betas = 1.0 - (alpha_bars[1:] / alpha_bars[:-1])
    
    # 4. Clip betas to a maximum value to maintain numerical stability
    betas = torch.clamp(betas, min=0.0, max=beta_max)
    
    # 5. Recompute alphas (1 - betas) and cumulative alpha_bars sequentially to stay consistent
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    
    return {
        'betas': betas,
        'alphas': alphas,
        'alpha_bars': alpha_bars
    }