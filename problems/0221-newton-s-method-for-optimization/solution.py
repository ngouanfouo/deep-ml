from typing import Callable
import numpy as np

def newtons_method_optimization(
    gradient_func: Callable[[list[float]], list[float]],
    hessian_func: Callable[[list[float]], list[list[float]]],
    x0: list[float],
    tol: float = 1e-6,
    max_iter: int = 100
) -> list[float]:
    """
    Find the minimum of a function using Newton's method.
    
    Args:
        gradient_func: Function that returns gradient vector at a point
        hessian_func: Function that returns Hessian matrix at a point
        x0: Initial guess (list of coordinates)
        tol: Convergence tolerance for gradient norm
        max_iter: Maximum number of iterations
        
    Returns:
        The point that minimizes the function
    """
    x = np.array(x0, dtype=np.float64)
    
    for iteration in range(max_iter):
        # Compute gradient and Hessian at current point
        grad = np.array(gradient_func(x.tolist()), dtype=np.float64)
        hessian = np.array(hessian_func(x.tolist()), dtype=np.float64)
        
        # Check convergence: if gradient norm is below tolerance, stop
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            break
        
        # Solve Newton system: H * delta = -grad
        # Use numpy's linear algebra solver
        try:
            delta = np.linalg.solve(hessian, -grad)
        except np.linalg.LinAlgError:
            # If Hessian is singular, use pseudo-inverse or fallback to gradient descent
            # This is a fallback; for well-behaved functions, Hessian should be invertible
            delta = -grad / (np.linalg.norm(grad) + 1e-10)
        
        # Update x
        x = x + delta
        
        # Optional: check if Hessian is positive definite
        # If not, Newton's method might not converge to a minimum
        # This is a simple check but could be enhanced
        
        # Optional: line search to ensure descent
        # For quadratic functions, step size 1 is optimal
    
    return x.tolist()