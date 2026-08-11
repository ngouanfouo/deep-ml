import torch

def quotient_rule_derivative(g_coeffs: list, h_coeffs: list, x: float) -> torch.Tensor:
    """
    Compute the derivative of f(x) = g(x)/h(x) at point x using the quotient rule.
    
    Args:
        g_coeffs: Coefficients of numerator polynomial in descending order
        h_coeffs: Coefficients of denominator polynomial in descending order
        x: Point at which to evaluate the derivative
        
    Returns:
        The derivative value f'(x) as a scalar torch.Tensor
    """
    # Convert coefficients to torch tensors
    g = torch.tensor(g_coeffs, dtype=torch.float32)
    h = torch.tensor(h_coeffs, dtype=torch.float32)
    
    # Evaluate g(x) and h(x) at point x
    # Polynomial evaluation: sum(coeff[i] * x^(n-1-i)) where n is degree
    def eval_poly(coeffs, x_val):
        n = len(coeffs)
        # Create powers of x: [x^(n-1), x^(n-2), ..., x^0]
        powers = torch.tensor([x_val ** (n - 1 - i) for i in range(n)], dtype=torch.float32)
        return torch.sum(coeffs * powers)
    
    g_x = eval_poly(g, x)
    h_x = eval_poly(h, x)
    
    # Compute derivatives of g and h
    # For coeffs in descending order [c0, c1, ..., cn] representing c0*x^n + c1*x^(n-1) + ... + cn
    # Derivative: [c0*n, c1*(n-1), ..., c(n-1)*1]
    def derivative_coeffs(coeffs):
        n = len(coeffs) - 1  # highest degree
        if n <= 0:
            return torch.tensor([], dtype=torch.float32)
        # Multiply each coefficient by its power, excluding the constant term
        degrees = torch.tensor([n - i for i in range(len(coeffs))], dtype=torch.float32)
        derivative = coeffs[:-1] * degrees[:-1]  # Exclude constant term
        return derivative
    
    g_prime_coeffs = derivative_coeffs(g)
    h_prime_coeffs = derivative_coeffs(h)
    
    # Evaluate g'(x) and h'(x)
    g_prime_x = eval_poly(g_prime_coeffs, x) if len(g_prime_coeffs) > 0 else torch.tensor(0.0)
    h_prime_x = eval_poly(h_prime_coeffs, x) if len(h_prime_coeffs) > 0 else torch.tensor(0.0)
    
    # Apply quotient rule: f' = (g' * h - g * h') / h^2
    numerator = g_prime_x * h_x - g_x * h_prime_x
    denominator = h_x ** 2
    
    result = numerator / denominator
    
    return result