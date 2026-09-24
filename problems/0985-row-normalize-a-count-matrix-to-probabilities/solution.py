import numpy as np

def row_normalize(counts: list[list[float]]) -> list[list[float]]:
    """Convert a count matrix into a row-stochastic probability matrix."""
    counts = np.asarray(counts, dtype=float)
    
    # Row sums as a column vector of shape (n, 1)
    row_sums = counts.sum(axis=1, keepdims=True)
    
    # Prepare output array; rows with zero sum remain all zeros
    probs = np.zeros_like(counts, dtype=float)
    
    # Divide only where row_sum != 0; broadcasting works because row_sums is (n, 1)
    np.divide(counts, row_sums, out=probs, where=row_sums != 0)
    
    return probs.tolist()