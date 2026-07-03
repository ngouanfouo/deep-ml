import torch

def gauss_seidel(A: torch.Tensor, b: torch.Tensor, n: int, x_ini=None) -> torch.Tensor:
    """
    Implements the Gauss-Seidel iterative method for solving linear systems Ax = b.

    Args:
        A: Square coefficient matrix (torch.Tensor)
        b: Right-hand side vector (torch.Tensor)
        n: Number of iterations
        x_ini: Optional initial guess tensor (if None, zeros are used)

    Returns:
        Approximated solution vector x after n iterations
    """
    # Get the size of the system
    m = A.shape[0]
    
    # Initialize x with zeros if no initial guess is provided
    if x_ini is None:
        x = torch.zeros(m, dtype=A.dtype)
    else:
        x = x_ini.clone()
    
    # Gauss-Seidel iteration
    for _ in range(n):
        for i in range(m):
            # Compute sum of A[i][j] * x[j] for j != i
            sum1 = torch.dot(A[i, :i], x[:i])
            sum2 = torch.dot(A[i, i+1:], x[i+1:])
            
            # Update x[i] using the most recent values
            x[i] = (b[i] - sum1 - sum2) / A[i, i]
    
    return x