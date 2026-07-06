import numpy as np

def prelu_forward(x: np.ndarray, alpha: float = 0.25) -> np.ndarray:
    """
    Implements the forward pass of PReLU.
    Args:
        x: Input array of any shape
        alpha: Slope parameter for negative values (default: 0.25)
    Returns:
        np.ndarray: PReLU activation output, same shape as x
    """
    return np.where(x > 0, x, alpha * x)


def prelu_backward(x: np.ndarray, alpha: float, grad_output: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Implements the backward pass of PReLU, computing gradients for both x and alpha.
    Args:
        x: Original input from forward pass
        alpha: Slope parameter used in forward pass
        grad_output: Upstream gradient, same shape as x
    Returns:
        grad_x: Gradient w.r.t. input x, same shape as x
        grad_alpha: Gradient w.r.t. alpha (scalar, summed over all elements)
    """
    # grad_x is 1 where x > 0, and alpha where x <= 0, multiplied by the upstream gradient
    grad_x = np.where(x > 0, grad_output, grad_output * alpha)
    
    # grad_alpha is the sum of grad_output * x where x <= 0
    grad_alpha = float(np.sum(np.where(x <= 0, grad_output * x, 0.0)))
    
    return grad_x, grad_alpha