import torch

def lu_decomposition(A) -> tuple:
    """
    Perform LU decomposition on a square matrix using Doolittle's algorithm.
    
    Args:
        A: Square matrix as a list of lists, numpy array, or PyTorch tensor.
        
    Returns:
        tuple: (L, U) where L and U are torch.Tensor objects such that A = L @ U.
    """
    # Convert input to PyTorch float64 tensor
    if not isinstance(A, torch.Tensor):
        A_tensor = torch.tensor(A, dtype=torch.float64)
    else:
        A_tensor = A.to(dtype=torch.float64)
        
    n = A_tensor.shape[0]
    
    # Initialize L as Identity matrix and U as Zero matrix
    L = torch.eye(n, dtype=torch.float64)
    U = torch.zeros((n, n), dtype=torch.float64)
    
    for i in range(n):
        # 1. Compute row i of U
        for j in range(i, n):
            U[i, j] = A_tensor[i, j] - torch.dot(L[i, :i], U[:i, j])
            
        # 2. Compute column i of L (below diagonal)
        for j in range(i + 1, n):
            L[j, i] = (A_tensor[j, i] - torch.dot(L[j, :i], U[:i, i])) / U[i, i]
            
    return L, U