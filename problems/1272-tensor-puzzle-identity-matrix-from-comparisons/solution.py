import numpy as np

def eye(n: int) -> np.ndarray:
    """Return the n x n identity matrix as float64."""
    return (np.arange(n)[:, np.newaxis] == np.arange(n)[np.newaxis, :]).astype(np.float64)