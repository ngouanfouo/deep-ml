import numpy as np


def acf(x: np.ndarray, nlags: int) -> np.ndarray:
    """Sample autocorrelation function.

    Args:
        x (np.ndarray): 1-D series.
        nlags (int): highest lag to compute.

    Returns:
        np.ndarray: length nlags + 1, entry 0 equal to 1.0.
    """
    x_arr = np.asarray(x, dtype=float)
    n = len(x_arr)
    if nlags < 0:
        raise ValueError("nlags must be non-negative")
    nlags = min(nlags, n - 1)
    
    x_c = x_arr - np.mean(x_arr)
    denom = np.sum(x_c ** 2)
    
    res = np.zeros(nlags + 1, dtype=float)
    res[0] = 1.0
    
    if denom == 0.0:
        return res
        
    for k in range(1, nlags + 1):
        num = np.sum(x_c[k:] * x_c[:-k])
        res[k] = num / denom
        
    return res


def ljung_box(x: np.ndarray, nlags: int) -> float:
    """Ljung-Box Q statistic over lags 1..nlags."""
    x_arr = np.asarray(x, dtype=float)
    n = len(x_arr)
    
    # Get autocorrelations from lag 1 to nlags
    rhos = acf(x_arr, nlags)[1:]
    k = np.arange(1, len(rhos) + 1)
    
    # Compute the portmanteau Q statistic
    terms = (rhos ** 2) / (n - k)
    q = n * (n + 2) * np.sum(terms)
    
    return float(q)