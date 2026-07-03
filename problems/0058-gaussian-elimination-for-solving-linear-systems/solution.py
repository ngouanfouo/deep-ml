import torch

def gaussian_elimination(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Solves the system Ax = b using Gaussian Elimination with partial pivoting.

    :param A: Coefficient matrix (torch.Tensor)
    :param b: Right-hand side vector (torch.Tensor)
    :return: Solution vector x (torch.Tensor)
    """
    # Convert inputs to float64 to ensure high numerical precision
    A = A.to(torch.float64).clone()
    b = b.to(torch.float64).clone()
    
    n = b.size(0)
    
    # --- Phase 1: Forward Elimination ---
    for i in range(n):
        # 1. Partial Pivoting
        pivot_row = i + torch.argmax(torch.abs(A[i:, i]))
        
        # Safe row swapping using a clone buffer
        if pivot_row != i:
            # Swap rows in A
            row_i_copy = A[i].clone()
            A[i] = A[pivot_row]
            A[pivot_row] = row_i_copy
            
            # Swap elements in b
            b_i_copy = b[i].clone()
            b[i] = b[pivot_row]
            b[pivot_row] = b_i_copy
            
        if torch.abs(A[i, i]) < 1e-12:
            raise ValueError("The matrix is singular or near-singular.")
            
        # 2. Elimination
        for j in range(i + 1, n):
            factor = A[j, i] / A[i, i]
            A[j, i:] -= factor * A[i, i:]
            b[j] -= factor * b[i]
            
    # --- Phase 2: Backward Substitution ---
    x = torch.zeros_like(b)
    for i in range(n - 1, -1, -1):
        sum_known = torch.dot(A[i, i + 1:], x[i + 1:])
        x[i] = (b[i] - sum_known) / A[i, i]
        
    return x