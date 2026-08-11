import torch

def classify_critical_point(hessian: torch.Tensor, tol: float = 1e-10):
    """
    Classify a critical point using Hessian eigenvalues.
    
    Args:
        hessian: A symmetric n x n torch.Tensor representing the Hessian matrix.
        tol: Tolerance for determining if an eigenvalue is effectively zero.
    
    Returns:
        -1 if all eigenvalues are strictly positive (local minimum)
        +1 if all eigenvalues are strictly negative (local maximum)
         0 if eigenvalues have mixed signs (saddle point)
        None if any eigenvalue is approximately zero (inconclusive)
    """
    # Input validation
    if hessian.numel() == 0:
        return None
    
    # Ensure Hessian is symmetric (optional but recommended)
    # If not symmetric, symmetrize it
    if not torch.allclose(hessian, hessian.T):
        hessian = (hessian + hessian.T) / 2
    
    # Compute eigenvalues
    eigenvalues = torch.linalg.eigvalsh(hessian)
    
    # Check if any eigenvalue is approximately zero
    if torch.any(torch.abs(eigenvalues) <= tol):
        return None
    
    # Check if all eigenvalues are strictly positive
    if torch.all(eigenvalues > tol):
        return -1  # local minimum
    
    # Check if all eigenvalues are strictly negative
    if torch.all(eigenvalues < -tol):
        return 1  # local maximum
    
    # If we reach here, eigenvalues have mixed signs
    return 0  # saddle point