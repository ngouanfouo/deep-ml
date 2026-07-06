import numpy as np

def orthonormal_basis(vectors: list[list[float]], tol: float = 1e-10) -> list[np.ndarray]:
    """
    Computes an orthonormal basis for the subspace spanned by a list of vectors
    using the Gram-Schmidt process.
    
    Args:
        vectors: A list of input vectors (each as a list of floats)
        tol: Tolerance threshold to detect and skip linearly dependent vectors
        
    Returns:
        list[np.ndarray]: A list of orthonormal numpy arrays spanning the same subspace.
    """
    basis = []
    
    for v_list in vectors:
        v = np.array(v_list, dtype=float)
        
        # Subtract the projections onto all previously found basis vectors
        for u in basis:
            projection = np.dot(v, u) * u
            v -= projection
            
        # Calculate the norm of the remaining orthogonal vector
        norm = np.linalg.norm(v)
        
        # If the norm is greater than the tolerance, it's linearly independent
        if norm > tol:
            basis.append(v / norm)
            
    return basis