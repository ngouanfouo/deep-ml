import torch

def ddpm_linear_schedule(beta_start: float, beta_end: float, T: int) -> dict:
    """
    Compute the DDPM linear noise schedule and derived quantities using PyTorch.

    Args:
        beta_start: Starting value of the beta (noise variance) schedule.
        beta_end: Ending value of the beta schedule.
        T: Number of diffusion timesteps.

    Returns:
        Dictionary with keys: 'betas', 'alphas', 'alpha_bars',
        'sqrt_alpha_bars', 'sqrt_one_minus_alpha_bars'.
        All values are torch.Tensor of length T.
    """
    # 1. Linearly spaced betas from beta_start to beta_end over T steps
    betas = torch.linspace(beta_start, beta_end, steps=T)
    
    # 2. Alphas: 1 - beta for each timestep
    alphas = 1.0 - betas
    
    # 3. Alpha bars: cumulative product of alphas
    alpha_bars = torch.cumprod(alphas, dim=0)
    
    # 4. Square root of alpha bars
    sqrt_alpha_bars = torch.sqrt(alpha_bars)
    
    # 5. Square root of (1 - alpha bars)
    sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)
    
    return {
        'betas': betas,
        'alphas': alphas,
        'alpha_bars': alpha_bars,
        'sqrt_alpha_bars': sqrt_alpha_bars,
        'sqrt_one_minus_alpha_bars': sqrt_one_minus_alpha_bars
    }