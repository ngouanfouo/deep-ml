import numpy as np

def to_categorical(x, n_col=None):
    # If n_col not provided, determine from max value + 1
    if n_col is None:
        n_col = np.max(x) + 1
    
    # Create one-hot encoded array
    n_samples = x.shape[0]
    one_hot = np.zeros((n_samples, n_col), dtype=np.float32)
    one_hot[np.arange(n_samples), x] = 1.0
    
    return one_hot