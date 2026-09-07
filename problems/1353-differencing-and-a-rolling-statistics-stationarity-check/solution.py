import numpy as np


def difference(x: np.ndarray, d: int) -> np.ndarray:
    """Apply first differencing d times.

    Args:
        x (np.ndarray): 1-D input series.
        d (int): number of differencing passes; 0 returns x unchanged.

    Returns:
        np.ndarray: series of length len(x) - d.
    """
    x_arr = np.asarray(x, dtype=float)
    if d == 0:
        return x_arr
    return np.diff(x_arr, n=d)


def stationarity_report(x: np.ndarray, n_chunks: int) -> tuple:
    """Return (mean_spread, std_ratio) across n_chunks equal contiguous chunks,
    each rounded to 4 decimals."""
    x_arr = np.asarray(x, dtype=float)
    n = len(x_arr)
    if n_chunks <= 0 or n == 0:
        return (0.0, 0.0)
    
    chunk_len = n // n_chunks
    if chunk_len == 0:
        chunk_len = 1
        n_chunks = n
        
    # Drop any remainder at the end
    x_trimmed = x_arr[:n_chunks * chunk_len]
    chunks = x_trimmed.reshape(n_chunks, chunk_len)
    
    chunk_means = np.mean(chunks, axis=1)
    chunk_stds = np.std(chunks, axis=1, ddof=0)
    
    mean_spread = round(float(np.max(chunk_means) - np.min(chunk_means)), 4)
    
    min_std = np.min(chunk_stds)
    if min_std == 0.0:
        if np.all(chunk_stds == 0.0):
            std_ratio = 1.0
        else:
            std_ratio = float('inf')
    else:
        std_ratio = round(float(np.max(chunk_stds) / min_std), 4)
        
    return (mean_spread, std_ratio)