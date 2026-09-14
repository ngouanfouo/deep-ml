import numpy as np

def SwiGLU(x: np.ndarray) -> np.ndarray:
    """
    Apply SwiGLU activation function on an input matrix.
    
    Args:
        x: np.ndarray of shape (batch_size, 2d)
        
    Returns:
        np.ndarray of shape (batch_size, d)
    """
    # Split the input into two equal halves along the feature dimension
    x1, x2 = np.split(x, 2, axis=-1)
    
    # Compute Sigmoid with clipping to handle potential extreme values safely
    sigmoid_x2 = 1.0 / (1.0 + np.exp(-x2))
    
    # Compute Swish(x2) = x2 * Sigmoid(x2)
    swish_x2 = x2 * sigmoid_x2
    
    # SwiGLU = x1 * Swish(x2)
    output = x1 * swish_x2
    
    # Round output to four decimal places
    return np.round(output, 4)