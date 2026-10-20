import numpy as np


def reverse_sde_step(x, t, dt, beta_fn, score_fn, rng):
    """
    One Euler-Maruyama step of the VP reverse SDE (Song et al. 2021, eq. 6) from t to t - dt.
    The noise z is a single rng.standard_normal(x.shape) draw. Returns the new (N, d) array.
    """
    x = np.asarray(x, dtype=float)
    beta = float(beta_fn(t))
    score = np.asarray(score_fn(x, t), dtype=float)

    drift = (0.5 * beta * x + beta * score) * dt
    noise = np.sqrt(beta * dt) * rng.standard_normal(x.shape)

    return x + drift + noise


def pf_ode_step(x, t, dt, beta_fn, score_fn):
    """
    One Euler step of the VP probability-flow ODE (eq. 13) from t to t - dt.
    Returns the new (N, d) array.
    """
    x = np.asarray(x, dtype=float)
    beta = float(beta_fn(t))
    score = np.asarray(score_fn(x, t), dtype=float)

    drift = (0.5 * beta * x + 0.5 * beta * score) * dt

    return x + drift


def sample_vp(x_T, ts, beta_fn, score_fn, mode="ode", rng=None):
    """
    Run the chosen sampler over a decreasing time grid ts (step i: t = ts[i], dt = ts[i] - ts[i+1]).
    mode is "ode" or "sde" (rng required for "sde"). Returns the final (N, d) array.
    """
    x = np.array(x_T, dtype=float, copy=True)
    ts = np.asarray(ts, dtype=float)

    for i in range(len(ts) - 1):
        t = float(ts[i])
        dt = float(ts[i] - ts[i + 1])
        if mode == "ode":
            x = pf_ode_step(x, t, dt, beta_fn, score_fn)
        elif mode == "sde":
            x = reverse_sde_step(x, t, dt, beta_fn, score_fn, rng)
        else:
            raise ValueError(f"unknown mode: {mode}")

    return x