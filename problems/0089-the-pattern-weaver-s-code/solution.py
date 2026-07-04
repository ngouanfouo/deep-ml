import numpy as np

def softmax(values):
    # Implement the softmax function
    # Subtract max for numerical stability
    exp_values = np.exp(values - np.max(values))
    return exp_values / np.sum(exp_values)

def pattern_weaver(n, crystal_values, dimension):
    # Your code here
    values = np.array(crystal_values)
    
    # Compute raw attention scores: dot product of each pair
    if dimension == 1:
        # For scalars, use outer product
        scores = np.outer(values, values)
    else:
        # For vectors, use dot product
        scores = values @ values.T
    
    # Apply softmax row-wise to get attention weights
    attention_weights = np.array([softmax(row) for row in scores])
    
    # Compute weighted pattern for each crystal
    weighted_pattern = attention_weights @ values
    
    # Round to 4 decimal places as shown in the example
    return np.round(weighted_pattern, 4)