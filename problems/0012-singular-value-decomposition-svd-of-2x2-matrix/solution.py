import torch

def svd_2x2_singular_values(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute SVD of a 2x2 matrix using one Jacobi rotation.
    
    Args:
        A: A 2x2 torch tensor
    
    Returns:
        Tuple (U, S, Vt) where A ≈ U @ diag(S) @ Vt
    """
    # Ensure input is float for math operations
    A = A.to(torch.float64)
    
    # 1. Compute ATA to find the right singular vectors (V)
    # ATA = [[a, b], [b, c]]
    ATA = A.T @ A
    a = ATA[0, 0]
    b = ATA[0, 1]
    c = ATA[1, 1]
    
    # 2. Calculate the Jacobi rotation angle theta to diagonalize ATA
    if torch.abs(b) < 1e-15:
        cos_theta = torch.tensor(1.0, dtype=torch.float64)
        sin_theta = torch.tensor(0.0, dtype=torch.float64)
    else:
        # Using the stable symmetric Jacobi formula: tan(2*theta) = 2*b / (a - c)
        tau = (a - c) / (2.0 * b)
        if tau >= 0:
            t = 1.0 / (tau + torch.sqrt(1.0 + tau**2))
        else:
            t = -1.0 / (-tau + torch.sqrt(1.0 + tau**2))
        
        cos_theta = 1.0 / torch.sqrt(1.0 + t**2)
        sin_theta = t * cos_theta

    # 3. Construct the Right Singular Vector Matrix V and Vt
    V = torch.tensor([
        [cos_theta, -sin_theta],
        [sin_theta,  cos_theta]
    ], dtype=torch.float64)
    Vt = V.T
    
    # 4. Compute B = A @ V. The columns of B are U * S
    B = A @ V
    
    # 5. Extract Singular Values (magnitudes of the columns of B)
    s0 = torch.linalg.norm(B[:, 0])
    s1 = torch.linalg.norm(B[:, 1])
    
    # 6. Normalize columns of B to get Left Singular Vectors U
    u0 = B[:, 0] / s0 if s0 > 1e-15 else torch.tensor([1.0, 0.0], dtype=torch.float64)
    u1 = B[:, 1] / s1 if s1 > 1e-15 else torch.tensor([0.0, 1.0], dtype=torch.float64)
    
    U = torch.stack((u0, u1), dim=1)
    S = torch.tensor([s0, s1], dtype=torch.float64)
    
    # 7. Sort singular values in descending order for standard SVD convention
    if S[0] < S[1]:
        S = S[[1, 0]]
        U = U[:, [1, 0]]
        Vt = Vt[[1, 0], :]

    return U, S, Vt


