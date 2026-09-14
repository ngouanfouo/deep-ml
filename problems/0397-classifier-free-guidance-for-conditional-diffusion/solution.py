import numpy as np

def classifier_free_guidance_step(
    eps_uncond: list,
    eps_cond: list,
    x_t: list,
    guidance_scale: float,
    alpha_bar_t: float,
    alpha_bar_t_minus_1: float,
    beta_t: float
) -> tuple:
    """
    Perform one reverse diffusion step with Classifier-Free Guidance.
    
    Args:
        eps_uncond: Unconditional noise prediction from the model
        eps_cond: Conditional noise prediction from the model
        x_t: Current noisy sample at timestep t
        guidance_scale: CFG weight (w). w=1 gives standard conditional, w>1 amplifies conditioning
        alpha_bar_t: Cumulative product of alpha up to timestep t
        alpha_bar_t_minus_1: Cumulative product of alpha up to timestep t-1
        beta_t: Noise schedule value at timestep t
    
    Returns:
        Tuple of (guided_eps, predicted_x0, posterior_mean) as lists rounded to 4 decimals
    """
    eps_u = np.array(eps_uncond, dtype=float)
    eps_c = np.array(eps_cond, dtype=float)
    xt = np.array(x_t, dtype=float)
    
    # 1. Compute the guided noise prediction: eps_g = eps_u + w * (eps_c - eps_u)
    guided_eps = eps_u + guidance_scale * (eps_c - eps_u)
    
    # 2. Estimate the clean sample x_0 from x_t and guided noise
    # x_0 = (x_t - sqrt(1 - alpha_bar_t) * guided_eps) / sqrt(alpha_bar_t)
    sqrt_alpha_bar_t = np.sqrt(alpha_bar_t)
    sqrt_one_minus_alpha_bar_t = np.sqrt(1.0 - alpha_bar_t)
    predicted_x0 = (xt - sqrt_one_minus_alpha_bar_t * guided_eps) / sqrt_alpha_bar_t
    
    # 3. Compute the posterior mean for the previous timestep using the standard DDPM formula
    # mean = (sqrt(alpha_bar_{t-1}) * beta_t / (1 - alpha_bar_t)) * x_0 + 
    #        (sqrt(alpha_t) * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)) * x_t
    alpha_t = 1.0 - beta_t
    sqrt_alpha_t = np.sqrt(alpha_t)
    sqrt_alpha_bar_prev = np.sqrt(alpha_bar_t_minus_1)
    
    coeff_x0 = (sqrt_alpha_bar_prev * beta_t) / (1.0 - alpha_bar_t)
    coeff_xt = (sqrt_alpha_t * (1.0 - alpha_bar_t_minus_1)) / (1.0 - alpha_bar_t)
    
    posterior_mean = coeff_x0 * predicted_x0 + coeff_xt * xt
    
    # Round results to 4 decimal places and convert to lists
    return (
        [round(val, 4) for val in guided_eps.tolist()],
        [round(val, 4) for val in predicted_x0.tolist()],
        [round(val, 4) for val in posterior_mean.tolist()]
    )