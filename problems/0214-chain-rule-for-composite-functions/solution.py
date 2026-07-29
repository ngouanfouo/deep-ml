import numpy as np

def compute_chain_rule_gradient(functions: list[str], x: float) -> float:
    """
    Compute derivative of composite functions using chain rule.
    
    Args:
        functions: List of function names (applied right to left)
                  Available: 'square', 'sin', 'exp', 'log'
        x: Point at which to evaluate derivative
    
    Returns:
        Derivative value at x
    
    Example:
        ['sin', 'square'] represents sin(x²)
        ['exp', 'sin', 'square'] represents exp(sin(x²))
    """
    def apply_function(func_name, value, compute_derivative=False):
        """Apply function or its derivative at given value."""
        if func_name == 'square':
            if compute_derivative:
                return 2 * value  # d/dx (x²) = 2x
            else:
                return value ** 2
        elif func_name == 'sin':
            if compute_derivative:
                return np.cos(value)  # d/dx (sin(x)) = cos(x)
            else:
                return np.sin(value)
        elif func_name == 'exp':
            if compute_derivative:
                return np.exp(value)  # d/dx (exp(x)) = exp(x)
            else:
                return np.exp(value)
        elif func_name == 'log':
            if compute_derivative:
                return 1.0 / value  # d/dx (log(x)) = 1/x
            else:
                return np.log(value)
        else:
            raise ValueError(f"Unknown function: {func_name}")
    
    # To compute h'(x) where h = f1 ∘ f2 ∘ ... ∘ fn
    # h'(x) = f1'(fn(...f2(f1(x))...)) * f2'(...) * ... * fn'(x)
    
    # First, compute all intermediate values: g_k(x) where g_k is composition
    # of functions from k to n (applied in order)
    # We want: f1'(g2(x)) * f2'(g3(x)) * ... * fn'(x)
    
    # Compute all intermediate compositions
    # value_at_depth[i] = value after applying functions from i to end
    n = len(functions)
    values = [x]  # values[0] = x
    
    # Compute values of compositions from inner to outer
    current = x
    for func_name in reversed(functions):
        current = apply_function(func_name, current, compute_derivative=False)
        values.append(current)
    
    # values[0] = x
    # values[1] = f_n(x) (innermost)
    # values[2] = f_{n-1}(f_n(x))
    # ...
    # values[n] = f_1(f_2(...f_n(x)...)) (full composition)
    
    # Now compute derivative using chain rule
    # h'(x) = f1'(values[n-1]) * f2'(values[n-2]) * ... * fn'(values[0])
    derivative = 1.0
    for i, func_name in enumerate(functions):
        # The derivative of function i (from outer to inner) should be evaluated
        # at the composition of all inner functions
        # For function i (0-indexed from outer), inner composition is values[n - i - 1]
        inner_value = values[n - i - 1]
        func_deriv = apply_function(func_name, inner_value, compute_derivative=True)
        derivative *= func_deriv
    
    return float(derivative)