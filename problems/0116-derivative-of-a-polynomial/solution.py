import torch

def poly_term_derivative(c: float, x: float, n: float) -> torch.Tensor:
    """
    Compute the derivative of a polynomial term c * x^n at point x.

    Args:
        c: coefficient of the term
        x: point at which to evaluate the derivative
        n: exponent of the term

    Returns:
        The value of the derivative at point x as a tensor
    """
    x_t = torch.tensor(x, dtype=torch.float32)

    # Power rule: d/dx (c * x^n) = c * n * x^(n-1)
    # Special case n = 0: derivative is 0.
    if n == 0:
        return torch.tensor(0.0, dtype=torch.float32)

    return c * n * (x_t ** (n - 1))