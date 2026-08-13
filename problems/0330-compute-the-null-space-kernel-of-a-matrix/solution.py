import torch

def compute_null_space(A: torch.Tensor, tol: float = 1e-10) -> torch.Tensor:
    """
    Compute an orthonormal basis for the null space using QR decomposition.
    
    Args:
        A: Input tensor of shape (m, n)
        tol: Tolerance for considering singular values as zero
    
    Returns:
        Tensor of shape (n, k) where k is the dimension of the null space.
        Columns form an orthonormal basis for the null space.
    """
    # Convert to double for numerical stability
    A = A.double()
    
    # Get dimensions
    m, n = A.shape
    
    # Handle empty matrix
    if m == 0 or n == 0:
        return torch.empty(n, 0, dtype=torch.float64)
    
    # For tall matrices, compute QR of A
    if m >= n:
        # Compute QR decomposition of A
        Q, R = torch.linalg.qr(A, mode='reduced')
        
        # Find the rank by examining diagonal of R
        diag_R = torch.abs(torch.diag(R))
        if diag_R.numel() > 0 and diag_R[0].item() > 0:
            svd_tol = max(m, n) * diag_R[0].item() * tol
        else:
            svd_tol = tol
        
        rank = torch.sum(diag_R > svd_tol).item()
        null_space_dim = n - rank
        
        if null_space_dim == 0:
            return torch.empty(n, 0, dtype=torch.float64)
        
        # For null space, we need the null space of R (upper triangular)
        # This is more involved and we'd need to solve Rx = 0
        # The SVD approach is simpler and more reliable
        
    # For wide matrices or general case, use SVD (more robust)
    U, s, Vt = torch.linalg.svd(A, full_matrices=True)
    
    # Calculate tolerance
    if s.numel() > 0 and s[0].item() > 0:
        svd_tol = max(m, n) * s[0].item() * tol
    else:
        svd_tol = tol
    
    rank = torch.sum(s > svd_tol).item()
    null_space_dim = n - rank
    
    if null_space_dim == 0:
        return torch.empty(n, 0, dtype=torch.float64)
    
    # Extract null space basis
    null_space_basis = Vt[rank:, :].T
    
    return null_space_basis