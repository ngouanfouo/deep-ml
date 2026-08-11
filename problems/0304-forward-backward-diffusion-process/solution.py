import numpy as np

def diffusion_process(x_0: np.ndarray, 
                      betas: np.ndarray,
                      timestep: int,
                      forward_noise: np.ndarray,
                      predicted_noise: np.ndarray,
                      backward_noise: np.ndarray = None) -> tuple:
    """
    Implement forward and backward diffusion processes.
    
    Args:
        x_0: Original clean data (any shape)
        betas: Noise schedule array of shape (T,)
        timestep: Current timestep t (1-indexed)
        forward_noise: Noise epsilon for forward diffusion
        predicted_noise: Model's predicted noise for backward diffusion
        backward_noise: Random noise z for stochastic backward step
    
    Returns:
        tuple: (x_t, x_t_minus_1) - noisy and denoised samples
    """
    # Calculate alphas and cumulative products
    alphas = 1 - betas
    alpha_bar = np.cumprod(alphas)
    
    # Get alpha_bar at timestep t (0-indexed)
    alpha_bar_t = alpha_bar[timestep - 1]
    
    # --- Forward Diffusion: x_t from x_0 ---
    # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
    sqrt_alpha_bar_t = np.sqrt(alpha_bar_t)
    sqrt_one_minus_alpha_bar_t = np.sqrt(1 - alpha_bar_t)
    x_t = sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * forward_noise
    
    # --- Backward Diffusion: x_{t-1} from x_t ---
    # Get alpha_t and beta_t
    alpha_t = alphas[timestep - 1]
    beta_t = betas[timestep - 1]
    
    # Get alpha_bar_{t-1} (for t=1, this would be 1.0)
    if timestep == 1:
        alpha_bar_t_minus_1 = 1.0
    else:
        alpha_bar_t_minus_1 = alpha_bar[timestep - 2]
    
    # Compute posterior distribution parameters
    # coef1 = 1 / sqrt(alpha_t)
    coef1 = 1 / np.sqrt(alpha_t)
    
    # coef2 = (1 - alpha_t) / sqrt(1 - alpha_bar_t)
    coef2 = (1 - alpha_t) / np.sqrt(1 - alpha_bar_t)
    
    # Compute mean of posterior distribution
    # mu = coef1 * (x_t - coef2 * predicted_noise)
    mu = coef1 * (x_t - coef2 * predicted_noise)
    
    # Compute posterior variance
    # sigma_t^2 = (1 - alpha_t) * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
    if timestep == 1:
        # For t=1, variance is 0 (no noise added in final step)
        sigma_sq = 0
    else:
        sigma_sq = (1 - alpha_t) * (1 - alpha_bar_t_minus_1) / (1 - alpha_bar_t)
    
    # Add noise for stochastic backward step (except when t=1)
    if timestep == 1:
        # No noise added in final step
        x_t_minus_1 = mu
    else:
        # Sample x_{t-1} from posterior distribution
        sigma = np.sqrt(sigma_sq)
        x_t_minus_1 = mu + sigma * backward_noise
    
    return x_t, x_t_minus_1