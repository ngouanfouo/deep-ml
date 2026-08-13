import torch

def cholesky_decomposition(A):
    """
    Perform Cholesky decomposition on a symmetric positive-definite matrix.
    
    Args:
        A: A symmetric positive-definite matrix (2D list, numpy array, or torch.Tensor)
    
    Returns:
        L: Lower triangular torch.Tensor such that A = L @ L.T,
           or -1 if decomposition is not possible.
    """
    # 1. Input validation & conversion to double-precision tensor
    if A is None:
        return -1

    try:
        if isinstance(A, torch.Tensor):
            A_tensor = A.clone().to(dtype=torch.float64)
        else:
            A_tensor = torch.tensor(A, dtype=torch.float64)
    except Exception:
        return -1

    # 2. Check dimensions (must be 2D non-empty square matrix)
    if A_tensor.ndim != 2:
        return -1
    
    n, m = A_tensor.shape
    if n == 0 or m == 0 or n != m:
        return -1

    # 3. Check symmetry: A == A^T
    if not torch.allclose(A_tensor, A_tensor.T, rtol=1e-5, atol=1e-6):
        return -1

    # 4. Compute Cholesky factor L
    L = torch.zeros((n, n), dtype=torch.float64)

    for i in range(n):
        for j in range(i + 1):
            s = torch.sum(L[i, :j] * L[j, :j])
            
            if i == j:
                val = A_tensor[i, i] - s
                # Positive-definiteness requires val > 0
                if val <= 0:
                    return -1
                L[i, j] = torch.sqrt(val)
            else:
                if L[j, j] == 0:
                    return -1
                L[i, j] = (A_tensor[i, j] - s) / L[j, j]

    return L