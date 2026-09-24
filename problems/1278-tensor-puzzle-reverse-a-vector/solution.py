import numpy as np

def flip(a: np.ndarray) -> np.ndarray:
    """Reverse 1-D array a without slicing a[::-1]."""
    n = a.shape[0]
    idx = n - 1 - np.arange(n)
    return a[idx]