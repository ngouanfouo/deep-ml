import numpy as np

def jacobian_matrix(f, x: list[float], h: float = 1e-5) -> list[list[float]]:
    """
    Compute the Jacobian matrix using numerical differentiation.
    
    Args:
        f: Function that takes a list of length n and returns a list of length m
        x: Point at which to evaluate the Jacobian (list of length n)
        h: Step size for central finite differences
    
    Returns:
        Jacobian matrix as a list of lists (m x n)
    """
    n = len(x)
    x_arr = np.array(x, dtype=np.float64)
    
    # 1. Determine the output dimension m by running a single probe evaluation
    probe_output = f(x)
    m = len(probe_output)
    
    # 2. Allocate space for the final m x n Jacobian structure
    jacobian = np.zeros((m, n), dtype=np.float64)
    
    # 3. Compute partial derivatives for each input dimension j using central difference
    for j in range(n):
        # Create small perturbation steps along the j-th coordinate axis
        x_plus = x_arr.copy()
        x_minus = x_arr.copy()
        
        x_plus[j] += h
        x_minus[j] -= h
        
        # Evaluate function variations
        f_plus = np.array(f(x_plus.tolist()), dtype=np.float64)
        f_minus = np.array(f(x_minus.tolist()), dtype=np.float64)
        
        # Central difference formula: df/dx_j ≈ (f(x + h) - f(x - h)) / (2 * h)
        jacobian[:, j] = (f_plus - f_minus) / (2.0 * h)
        
    return jacobian.tolist()