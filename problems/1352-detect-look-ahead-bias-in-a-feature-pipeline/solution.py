import numpy as np


def causal_zscore(x: np.ndarray, window: int) -> np.ndarray:
    """Rolling z-score using only the trailing `window` values at each point.

    Args:
        x (np.ndarray): 1-D input series.
        window (int): lookback length, inclusive of the current point.

    Returns:
        np.ndarray: same length as x; the first window-1 entries are np.nan.
    """
    n = len(x)
    res = np.full(n, np.nan, dtype=float)
    if window <= 0 or window > n:
        return res
    
    # Compute rolling means efficiently using cumsum
    padded_x = np.insert(x, 0, 0.0)
    cumsum_x = np.cumsum(padded_x)
    window_sums = cumsum_x[window:] - cumsum_x[:-window]
    rolling_mean = window_sums / window
    
    # Compute rolling variance (ddof=0) and population standard deviation
    padded_x2 = np.insert(x**2, 0, 0.0)
    cumsum_x2 = np.cumsum(padded_x2)
    window_sums_sq = cumsum_x2[window:] - cumsum_x2[:-window]
    
    rolling_var = (window_sums_sq / window) - rolling_mean**2
    rolling_var = np.clip(rolling_var, 0.0, None)  # Guard against floating-point underflow < 0
    rolling_std = np.sqrt(rolling_var)
    
    # Calculate z-score safely handling division by zero (e.g., constant windows)
    window_slice = x[window - 1:]
    z = np.zeros_like(window_slice, dtype=float)
    
    nonzero_std = rolling_std > 0
    z[nonzero_std] = (window_slice[nonzero_std] - rolling_mean[nonzero_std]) / rolling_std[nonzero_std]
    # When std is 0, z-score is 0.0 if value equals mean (which it does since std=0)
    
    res[window - 1:] = z
    return res


def has_lookahead(feature_fn, x: np.ndarray) -> bool:
    """Return True if feature_fn uses information from the future."""
    base = feature_fn(x)
    
    # Copy array and perturb the last element without mutating the original
    x_perturbed = x.copy()
    x_perturbed[-1] += 1000.0
    
    new = feature_fn(x_perturbed)
    
    # Check if any earlier outputs changed, treating NaN == NaN as equal
    return not np.allclose(base[:-1], new[:-1], equal_nan=True)