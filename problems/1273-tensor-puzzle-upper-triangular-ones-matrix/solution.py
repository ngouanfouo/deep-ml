import numpy as np

def triu_ones(n: int) -> np.ndarray:
    """n x n upper-triangular matrix of ones (including diagonal)."""
    return (np.arange(n)[:, None] <= np.arange(n)[None, :]).astype(np.float64)