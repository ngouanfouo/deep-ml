import numpy as np

def qr_decomposition(A: list[list[float]]) -> tuple[list[list[float]], list[list[float]]]:
    """
    Perform QR decomposition using Gram-Schmidt process.
    
    Args:
        A: An m x n matrix represented as list of lists
    
    Returns:
        Tuple of (Q, R) where Q is orthogonal and R is upper triangular
    """
    # Convert to numpy array for easier computation
    A = np.array(A, dtype=np.float64)
    m, n = A.shape
    
    # Initialize Q and R
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    
    # Gram-Schmidt process
    for j in range(n):
        # Start with the j-th column of A
        v = A[:, j].copy()
        
        # Subtract projections onto previous orthonormal vectors
        for i in range(j):
            # r_ij = q_i^T * a_j
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v = v - R[i, j] * Q[:, i]
        
        # r_jj = ||v||
        R[j, j] = np.linalg.norm(v)
        
        # q_j = v / ||v||
        if R[j, j] > 1e-10:
            Q[:, j] = v / R[j, j]
        else:
            # If v is zero, set q_j to zero vector
            Q[:, j] = np.zeros(m)
    
    # Convert back to list of lists
    return Q.tolist(), R.tolist()