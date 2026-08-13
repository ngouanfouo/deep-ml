import torch

def check_positive_definite(matrix: list) -> dict:
    """
    Check if a square matrix is positive definite and compute its eigenvalues.
    
    Args:
        matrix: A 2D list representing a square matrix
        
    Returns:
        dict with 'is_positive_definite' (bool) and 'eigenvalues' (list of floats sorted ascending)
    """
    # Convert to float64 tensor for numerical precision
    A = torch.tensor(matrix, dtype=torch.float64)
    
    # Ensure matrix is square
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square")
    
    # 1. Check symmetry (x^T A x > 0 requires a real symmetric matrix in real space)
    is_symmetric = bool(torch.allclose(A, A.T, rtol=1e-5, atol=1e-8))
    
    # 2. Compute eigenvalues
    eigenvalues = torch.linalg.eigvals(A)
    eigenvalues_real = eigenvalues.real
    eigenvalues_imag = eigenvalues.imag
    
    # 3. Sort eigenvalues ascending by real part
    sorted_real = torch.sort(eigenvalues_real)[0]
    
    # 4. Numerical conditions for positive definiteness
    tolerance = 1e-10
    all_real = bool(torch.all(torch.abs(eigenvalues_imag) < tolerance).item())
    all_positive = bool(torch.all(sorted_real > tolerance).item())
    
    # Positive definite if symmetric, real eigenvalues, and all eigenvalues > 1e-10
    is_positive_definite = is_symmetric and all_real and all_positive
    
    # Round eigenvalues to 4 decimal places and convert to native floats
    eigenvalues_rounded = []
    for val in sorted_real:
        r_val = round(val.item(), 4)
        if r_val == 0.0:
            r_val = 0.0  # Avoid -0.0 representation
        eigenvalues_rounded.append(r_val)
    
    return {
        'is_positive_definite': is_positive_definite,
        'eigenvalues': eigenvalues_rounded
    }