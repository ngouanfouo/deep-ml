import numpy as np

def eigen_2x2_symmetric(M: np.ndarray) -> tuple:
    """Analytical eigendecomposition for a 2x2 symmetric matrix."""
    a, b = M[0, 0], M[0, 1]
    c = M[1, 1]
    
    trace = a + c
    det = a * c - b * b
    
    discriminant = max(0.0, trace**2 - 4 * det)
    sqrt_disc = np.sqrt(discriminant)
    
    lambda1 = (trace + sqrt_disc) / 2.0
    lambda2 = (trace - sqrt_disc) / 2.0
    
    if lambda1 < lambda2:
        lambda1, lambda2 = lambda2, lambda1
        
    def get_eigenvector(lam):
        if abs(b) > 1e-12:
            v = np.array([-b, a - lam])
        else:
            if abs(a - lam) < abs(c - lam):
                v = np.array([1.0, 0.0])
            else:
                v = np.array([0.0, 1.0])
                
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else np.array([1.0, 0.0])

    v1 = get_eigenvector(lambda1)
    v2 = get_eigenvector(lambda2)
    
    eigenvalues = np.array([lambda1, lambda2])
    eigenvectors = np.column_stack((v1, v2))
    
    return eigenvalues, eigenvectors

def svd_2x2(A: np.ndarray) -> tuple:
    """
    Compute SVD of a 2x2 matrix from fundamental principles.
    Returns U, s, and VT such that A = U @ diag(s) @ VT
    """
    A = A.astype(float)
    
    # 1. Compute V from A^T @ A
    ATA = A.T @ A
    s_squared, V = eigen_2x2_symmetric(ATA)
    
    s = np.sqrt(np.maximum(0.0, s_squared))
    
    # 2. Compute U from A @ A^T
    AAT = A @ A.T
    _, U = eigen_2x2_symmetric(AAT)
    
    # 3. Fix signs of U columns relative to V to satisfy A = U @ diag(s) @ V.T
    for i in range(2):
        if s[i] > 1e-9:
            predicted_Av = A @ V[:, i]
            actual_su = s[i] * U[:, i]
            if np.dot(predicted_Av, actual_su) < 0:
                U[:, i] = -U[:, i]
        else:
            if i == 1:
                if np.linalg.det(U) * np.linalg.det(V) < 0:
                    U[:, i] = -U[:, i]

    # Return V.T as the third parameter to perfectly match your verification harness
    return U, s, V.T

