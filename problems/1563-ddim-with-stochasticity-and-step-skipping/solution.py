import numpy as np


def ddim_timesteps(T, S, spacing="linear"):
    """
    S increasing integer timesteps in [0, T-1], starting at 0 and ending at T-1.
    "linear": np.linspace(0, T-1, S) rounded; "quadratic": np.linspace(0, sqrt(T-1), S)**2 rounded.
    Returns a Python list of ints. Raise ValueError for other spacings.
    """
    if spacing == "linear":
        ts = np.linspace(0, T - 1, S)
    elif spacing == "quadratic":
        ts = np.linspace(0, np.sqrt(T - 1), S) ** 2
    else:
        raise ValueError(f"unknown spacing: {spacing}")
    return [int(round(float(v))) for v in ts]


def ddim_sigma(a_t, a_prev, eta):
    """
    sigma = eta * sqrt((1 - a_prev) / (1 - a_t)) * sqrt(1 - a_t / a_prev)
    """
    inner = max(1.0 - a_t / a_prev, 0.0)
    return float(eta * np.sqrt((1.0 - a_prev) / (1.0 - a_t)) * np.sqrt(inner))


def ddim_step(x_t, eps, a_t, a_prev, eta, rng=None):
    """
    General DDIM update (Song et al. 2021, eq. 12).
    Draw z = rng.standard_normal(x_t.shape) only when sigma > 0.
    """
    x_t = np.asarray(x_t, dtype=float)
    eps = np.asarray(eps, dtype=float)

    sigma = ddim_sigma(a_t, a_prev, eta)

    x0_pred = (x_t - np.sqrt(1.0 - a_t) * eps) / np.sqrt(a_t)
    dir_coef = np.sqrt(max(1.0 - a_prev - sigma * sigma, 0.0))

    x_prev = np.sqrt(a_prev) * x0_pred + dir_coef * eps

    if sigma > 0:
        z = rng.standard_normal(x_t.shape)
        x_prev = x_prev + sigma * z

    return x_prev


def ddim_sample(x_T, eps_model, alpha_bar, timesteps, eta, rng=None):
    """
    Run ddim_step over the timesteps from largest to smallest.
    a_t = alpha_bar[t]; a_prev = alpha_bar[next smaller timestep], or 1.0 after the smallest one.
    eps_model(x, t) is called exactly once per timestep. Returns the final array.
    """
    alpha_bar = np.asarray(alpha_bar, dtype=float)
    x = np.asarray(x_T, dtype=float).copy()

    n = len(timesteps)
    for i in range(n - 1, -1, -1):
        t = timesteps[i]
        t_prev = timesteps[i - 1] if i > 0 else None

        a_t = alpha_bar[t]
        a_prev = alpha_bar[t_prev] if t_prev is not None else 1.0

        eps = eps_model(x, t)
        x = ddim_step(x, eps, a_t, a_prev, eta, rng=rng)

    return x