import torch
import torch.nn.functional as F

def product_rule_derivative(f_coeffs: list, g_coeffs: list) -> torch.Tensor:
    """
    Compute the derivative of the product of two polynomials.
    
    Args:
        f_coeffs: Coefficients of polynomial f, where f_coeffs[i] is the coefficient of x^i
        g_coeffs: Coefficients of polynomial g, where g_coeffs[i] is the coefficient of x^i
    
    Returns:
        torch.Tensor of coefficients of (f*g)' rounded to 4 decimal places
    """
    # Convert to torch tensors
    f = torch.tensor(f_coeffs, dtype=torch.float32)
    g = torch.tensor(g_coeffs, dtype=torch.float32)
    
    # Compute product f * g using polynomial multiplication (convolution)
    # Polynomial multiplication: (f*g)[k] = sum_i f[i] * g[k-i]
    # We'll implement this manually to avoid conv1d issues
    if len(f) == 0 or len(g) == 0:
        return torch.tensor([0.0])
    
    # Compute convolution manually
    product_len = len(f) + len(g) - 1
    product = torch.zeros(product_len, dtype=torch.float32)
    
    for i, f_coef in enumerate(f):
        for j, g_coef in enumerate(g):
            product[i + j] += f_coef * g_coef
    
    # Compute derivative of product: (f*g)' = sum(k * product[k] * x^(k-1))
    if product_len <= 1:
        derivative = torch.tensor([0.0])
    else:
        derivative = product[1:] * torch.arange(1, product_len, dtype=torch.float32)
    
    # Round to 4 decimal places
    derivative = torch.round(derivative * 10000) / 10000
    
    # Remove trailing zeros
    while len(derivative) > 1 and torch.abs(derivative[-1]) < 1e-6:
        derivative = derivative[:-1]
    
    # If result is empty or all zeros, return [0.0]
    if len(derivative) == 0 or torch.all(torch.abs(derivative) < 1e-6):
        return torch.tensor([0.0])
    
    return derivative