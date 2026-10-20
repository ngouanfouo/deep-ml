import numpy as np


def _alphas(betas):
    betas = np.asarray(betas, dtype=float)
    alphas = 1.0 - betas
    alpha_bars = np.cumprod(alphas)  # alpha_bars[t-1] = alpha_bar_t
    return betas, alphas, alpha_bars


def ddpm_posterior(x0, xt, t, betas):
    """
    Mean and variance of q(x_{t-1} | x_t, x_0) (Ho et al. 2020, eqs. 6-7).

    x0, xt: arrays of the same shape
    t:      1-indexed timestep in 1..T
    betas:  1-D array of length T (betas[t-1] is beta_t)

    Returns (mean, var) with mean an array like x0 and var a Python float.
    """
    betas, alphas, alpha_bars = _alphas(betas)
    x0 = np.asarray(x0, dtype=float)
    xt = np.asarray(xt, dtype=float)

    beta_t = betas[t - 1]
    alpha_t = alphas[t - 1]
    alpha_bar_t = alpha_bars[t - 1]
    alpha_bar_prev = alpha_bars[t - 2] if t >= 2 else 1.0

    coef_x0 = np.sqrt(alpha_bar_prev) * beta_t / (1.0 - alpha_bar_t)
    coef_xt = np.sqrt(alpha_t) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)

    mean = coef_x0 * x0 + coef_xt * xt

    tilde_beta = (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t) * beta_t
    return mean, float(tilde_beta)


def per_step_kl(x0, xt, t, betas, model_mean, decimals=6):
    """
    The L_{t-1} term: KL between the forward posterior and a model Gaussian with
    mean model_mean and variance tilde_beta_t, plus the equivalent weighted epsilon-MSE (eq. 12).
    Requires t >= 2.

    Returns a dict with keys "posterior_mean", "posterior_var", "kl", "eps_true",
    "eps_model", "weighted_eps_mse", all rounded to `decimals`.
    """
    betas, alphas, alpha_bars = _alphas(betas)
    x0 = np.asarray(x0, dtype=float)
    xt = np.asarray(xt, dtype=float)
    model_mean = np.asarray(model_mean, dtype=float)

    posterior_mean, posterior_var = ddpm_posterior(x0, xt, t, betas)

    beta_t = betas[t - 1]
    alpha_t = alphas[t - 1]
    alpha_bar_t = alpha_bars[t - 1]

    # KL between two isotropic Gaussians with the same variance: ||mu1 - mu2||^2 / (2 var).
    diff = posterior_mean - model_mean
    kl = float(np.sum(diff * diff) / (2.0 * posterior_var))

    # True noise that produced xt from x0.
    eps_true = (xt - np.sqrt(alpha_bar_t) * x0) / np.sqrt(1.0 - alpha_bar_t)

    # Model's implied noise, inverting eq. 11:
    #   mu_theta = (1/sqrt(alpha_t)) * (xt - beta_t/sqrt(1-alpha_bar_t) * eps_theta)
    eps_model = (xt - np.sqrt(alpha_t) * model_mean) * np.sqrt(1.0 - alpha_bar_t) / beta_t

    # Weighted epsilon-MSE, which equals the KL above.
    weight = (beta_t * beta_t) / (
        2.0 * posterior_var * alpha_t * (1.0 - alpha_bar_t)
    )
    weighted_eps_mse = float(weight * np.sum((eps_true - eps_model) ** 2))

    r = lambda a: np.round(np.asarray(a, dtype=float), decimals).tolist()

    return {
        "posterior_mean": r(posterior_mean),
        "posterior_var": round(float(posterior_var), decimals),
        "kl": round(kl, decimals),
        "eps_true": r(eps_true),
        "eps_model": r(eps_model),
        "weighted_eps_mse": round(weighted_eps_mse, decimals),
    }