import numpy as np

def unipc_step(
    x_t: np.ndarray,
    eps_t: np.ndarray,
    eps_prev: np.ndarray,
    alpha_t: float,
    sigma_t: float,
    alpha_prev: float,
    sigma_prev: float
) -> tuple:
    """
    Perform one UniPC predictor-corrector step.

    Args:
        x_t:       noisy sample at current step
        eps_t:     noise prediction at current step
        eps_prev:  noise prediction at previous step (reused)
        alpha_t:   alpha at current step
        sigma_t:   sigma at current step
        alpha_prev: alpha at target step
        sigma_prev: sigma at target step

    Returns:
        (x_predictor, x_corrector): tuple of arrays same shape as x_t
    """
    # 1) Estimate the clean sample x0 from the current noise prediction
    x0_pred = (x_t - sigma_t * eps_t) / alpha_t

    # 2) Predictor: standard DDIM step — project x0_pred to the target noise level
    x_predictor = alpha_prev * x0_pred + sigma_prev * eps_t

    # 3) Corrector: blend current and previous noise predictions, then reproject
    eps_blended = (eps_t + eps_prev) / 2.0
    x_corrector = alpha_prev * x0_pred + sigma_prev * eps_blended

    return x_predictor, x_corrector