from typing import Callable
import numpy as np

def compute_hessian(f: Callable[[list[float]], float], point: list[float], h: float = 1e-5) -> list[list[float]]:
    """
    Compute the Hessian matrix of function f at the given point using finite differences.
    
    Args:
        f: A scalar function that takes a list of floats and returns a float
        point: The point at which to compute the Hessian (list of coordinates)
        h: Step size for finite differences (default: 1e-5)
        
    Returns:
        The Hessian matrix as a list of lists (n x n where n = len(point))
    """
    n = len(point)
    point = np.array(point, dtype=np.float64)
    hessian = np.zeros((n, n), dtype=np.float64)
    
    # Compute gradient at the point using central differences (for the diagonal)
    # Actually, we'll compute second derivatives using central finite differences
    
    # For each pair (i, j), compute ∂²f/∂x_i∂x_j
    for i in range(n):
        for j in range(n):
            # Use central difference for mixed partial derivatives
            # ∂²f/∂x_i∂x_j ≈ (f(x + h*e_i + h*e_j) - f(x + h*e_i - h*e_j) 
            #                  - f(x - h*e_i + h*e_j) + f(x - h*e_i - h*e_j)) / (4*h²)
            
            # Create unit vectors
            e_i = np.zeros(n)
            e_j = np.zeros(n)
            e_i[i] = 1.0
            e_j[j] = 1.0
            
            # Evaluate f at four points
            f_pp = f((point + h * e_i + h * e_j).tolist())
            f_pm = f((point + h * e_i - h * e_j).tolist())
            f_mp = f((point - h * e_i + h * e_j).tolist())
            f_mm = f((point - h * e_i - h * e_j).tolist())
            
            # Central difference formula for second derivative
            hessian[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4.0 * h * h)
    
    # The Hessian should be symmetric, but due to numerical errors it might not be exactly
    # Average with its transpose to enforce symmetry
    hessian = (hessian + hessian.T) / 2.0
    
    return hessian.tolist()