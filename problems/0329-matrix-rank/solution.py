import torch

def matrix_rank(A: torch.Tensor, tol: float = 1e-10) -> int:
    """
    Compute the rank of a matrix using SVD.
    
    Args:
        A: Input matrix of shape (m, n) as a torch.Tensor
        tol: Tolerance for considering values as zero
    
    Returns:
        The rank of the matrix (integer)
    """
    # Convert to double for numerical stability
    A = A.double()
    
    # Handle empty matrix
    if A.numel() == 0:
        return 0
    
    # Compute SVD
    U, s, Vt = torch.linalg.svd(A, full_matrices=False)
    
    # Calculate tolerance based on the largest singular value
    if s.numel() > 0 and s[0].item() > 0:
        svd_tol = max(A.shape) * s[0].item() * tol
    else:
        # For zero matrix, all singular values are zero
        svd_tol = tol
    
    # Count singular values above tolerance
    rank = torch.sum(s > svd_tol).item()
    
    return rank