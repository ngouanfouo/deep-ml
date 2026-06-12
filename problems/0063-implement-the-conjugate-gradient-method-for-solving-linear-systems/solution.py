import torch

def conjugate_gradient(A: torch.Tensor, b: torch.Tensor, n: int, x0: torch.Tensor = None, tol: float = 1e-8) -> torch.Tensor:
    """
    Solve the system Ax = b using the Conjugate Gradient method.

    :param A: Symmetric positive-definite matrix (torch.Tensor)
    :param b: Right-hand side vector (torch.Tensor)
    :param n: Maximum number of iterations
    :param x0: Initial guess for solution (default is zero vector)
    :param tol: Convergence tolerance
    :return: Solution vector x as torch.Tensor
    """
    # Force computations in float64 for absolute numerical stability
    A = A.to(dtype=torch.float64)
    b = b.to(dtype=torch.float64)
    
    # 1. Initialize x0 guess vector
    if x0 is None:
        x = torch.zeros_like(b, dtype=torch.float64)
    else:
        x = x0.to(dtype=torch.float64)
        
    # 2. Initial residual evaluation: r0 = b - A @ x0
    r = b - (A @ x)
    
    # If the initial guess is already exceptionally close, break early
    r_dot_old = torch.dot(r, r)
    if torch.sqrt(r_dot_old) < tol:
        return x
        
    # First search direction is simply the initial residual direction
    p = r.clone()
    
    # 3. Iterative optimization loop
    for _ in range(n):
        Ap = A @ p
        
        # Compute step length alpha = (r^T * r) / (p^T * A * p)
        alpha = r_dot_old / torch.dot(p, Ap)
        
        # Update solution vector approximation
        x = x + alpha * p
        
        # Update residual vector
        r = r - alpha * Ap
        
        # Check convergence threshold metric
        r_dot_new = torch.dot(r, r)
        if torch.sqrt(r_dot_new) < tol:
            break
            
        # Compute conjugate projection adjustment scaler beta
        beta = r_dot_new / r_dot_old
        
        # Determine the next A-orthogonal search direction
        p = r + beta * p
        
        # Cycle variable records
        r_dot_old = r_dot_new

    return x
