import numpy as np


def poisson_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    """Poisson deviance, using the convention 0 * log(0) = 0."""
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)

    term = np.zeros_like(y, dtype=float)

    # Only evaluate log where y > 0, avoiding log(0).
    mask = y > 0
    term[mask] = y[mask] * np.log(y[mask] / mu[mask])

    dev = 2.0 * np.sum(term - (y - mu))
    return float(dev)


def dispersion_ratio(y: np.ndarray, mu: np.ndarray, n_params: int) -> float:
    """Pearson chi-square divided by (n - n_params)."""
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)

    n = y.size
    pearson_chi2 = np.sum((y - mu) ** 2 / mu)

    return float(pearson_chi2 / (n - n_params))