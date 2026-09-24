import numpy as np

def grassmann_distance(U_A: list[list[float]], U_B: list[list[float]]) -> float:
    """
    Compute the projection-based Grassmann distance between two subspaces
    represented by column-orthonormal matrices U_A and U_B.
    """
    U_A = np.asarray(U_A, dtype=float)
    U_B = np.asarray(U_B, dtype=float)

    # Cross-product whose singular values are cosines of principal angles
    M = U_A.T @ U_B
    S = np.linalg.svd(M, compute_uv=False)

    p = min(U_A.shape[1], U_B.shape[1])
    sum_cos_sq = np.sum(S ** 2)

    # Projection-based Grassmann distance = sqrt( sum sin^2(theta_i) )
    # = sqrt( p - sum cos^2(theta_i) )
    dist_sq = max(p - sum_cos_sq, 0.0)
    return float(np.sqrt(dist_sq))