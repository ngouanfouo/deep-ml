import numpy as np

def compute_partial_derivatives(func_name: str, point: tuple[float, ...]) -> tuple[float, ...]:
    """
    Compute partial derivatives of multivariable functions.
    
    Args:
        func_name: Function identifier
            'poly2d': f(x,y) = x²y + xy²
            'exp_sum': f(x,y) = e^(x+y)
            'product_sin': f(x,y) = x·sin(y)
            'poly3d': f(x,y,z) = x²y + yz²
            'squared_error': f(x,y) = (x-y)²
        point: Point (x, y) or (x, y, z) at which to evaluate
    
    Returns:
        Tuple of partial derivatives (∂f/∂x, ∂f/∂y, ...) at point
    """
    # Extract point values
    if len(point) == 2:
        x, y = point[0], point[1]
    elif len(point) == 3:
        x, y, z = point[0], point[1], point[2]
    else:
        raise ValueError("Point must be 2D or 3D")
    
    if func_name == 'poly2d':
        # f(x,y) = x²y + xy²
        # ∂f/∂x = 2xy + y²
        # ∂f/∂y = x² + 2xy
        df_dx = 2 * x * y + y ** 2
        df_dy = x ** 2 + 2 * x * y
        return (float(df_dx), float(df_dy))
    
    elif func_name == 'exp_sum':
        # f(x,y) = e^(x+y)
        # ∂f/∂x = e^(x+y)
        # ∂f/∂y = e^(x+y)
        value = np.exp(x + y)
        return (float(value), float(value))
    
    elif func_name == 'product_sin':
        # f(x,y) = x·sin(y)
        # ∂f/∂x = sin(y)
        # ∂f/∂y = x·cos(y)
        df_dx = np.sin(y)
        df_dy = x * np.cos(y)
        return (float(df_dx), float(df_dy))
    
    elif func_name == 'poly3d':
        # f(x,y,z) = x²y + yz²
        # ∂f/∂x = 2xy
        # ∂f/∂y = x² + z²
        # ∂f/∂z = 2yz
        df_dx = 2 * x * y
        df_dy = x ** 2 + z ** 2
        df_dz = 2 * y * z
        return (float(df_dx), float(df_dy), float(df_dz))
    
    elif func_name == 'squared_error':
        # f(x,y) = (x-y)²
        # ∂f/∂x = 2(x-y)
        # ∂f/∂y = -2(x-y)
        diff = x - y
        df_dx = 2 * diff
        df_dy = -2 * diff
        return (float(df_dx), float(df_dy))
    
    else:
        raise ValueError(f"Unknown function name: {func_name}")