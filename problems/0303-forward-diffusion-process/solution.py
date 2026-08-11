import numpy as np

def forward_diffusion(x_0: np.ndarray, t: int, beta_start: float, beta_end: float, num_timesteps: int, noise: np.ndarray) -> np.ndarray:
    """
    Apply forward diffusion process to add noise to input data.
    
    Args:
        x_0: Original input data (numpy array)
        t: Timestep (1-indexed, from 1 to num_timesteps)
        beta_start: Starting value of linear beta schedule
        beta_end: Ending value of linear beta schedule
        num_timesteps: Total number of diffusion timesteps
        noise: Noise array (same shape as x_0)
    
    Returns:
        Noisy sample x_t as numpy array
    """
    # Create linear beta schedule
    betas = np.linspace(beta_start, beta_end, num_timesteps)
    
    # Compute alphas
    alphas = 1 - betas
    
    # Compute cumulative product of alphas (alpha_bar)
    alpha_bar = np.cumprod(alphas)
    
    # Get alpha_bar for timestep t (adjust for 0-indexing)
    alpha_bar_t = alpha_bar[t - 1]
    
    # Compute coefficients for forward process
    # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
    sqrt_alpha_bar_t = np.sqrt(alpha_bar_t)
    sqrt_one_minus_alpha_bar_t = np.sqrt(1 - alpha_bar_t)
    
    # Compute noisy sample
    x_t = sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise
    
    return x_t