import numpy as np

def vector_sum(a: np.ndarray):
    """Sum elements of 1-D array a without np.sum / loops."""
    # Create a ones vector of the same length as a
    # Using np.arange and arithmetic (no np.ones allowed)
    n = len(a)
    ones = np.arange(n) * 0 + 1
    
    # Compute dot product: sum of all elements
    # np.dot computes the dot product between two arrays
    result = np.dot(a, ones)
    
    return result