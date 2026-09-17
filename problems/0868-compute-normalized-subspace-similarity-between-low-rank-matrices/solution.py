import numpy as np

def subspace_similarity(A: list, B: list, i: int, j: int) -> float:
    """
    Compute the normalized subspace similarity between the top-i left singular
    subspace of A and the top-j left singular subspace of B.

    Returns a float in [0, 1].
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    # Left singular vectors (columns of U)
    U_A, _, _ = np.linalg.svd(A, full_matrices=False)
    U_B, _, _ = np.linalg.svd(B, full_matrices=False)

    # Top-i and top-j left singular subspaces
    U_A_i = U_A[:, :i]   # shape: (rows, i)
    U_B_j = U_B[:, :j]   # shape: (rows, j)

    # Inner-product matrix between the two subspaces
    M = U_A_i.T @ U_B_j  # shape: (i, j)

    # Squared Frobenius norm = sum of squared singular values of M
    frob_sq = np.linalg.norm(M, 'fro') ** 2

    score = frob_sq / min(i, j)

    # Clamp for numerical safety (theoretical range is [0, 1])
    return float(np.clip(score, 0.0, 1.0))