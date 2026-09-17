import numpy as np

def projection_amplification(W: list[list[float]], delta_W: list[list[float]], r: int) -> float:
    """
    Compute the amplification factor of delta_W with respect to W
    using the top-r singular subspaces of delta_W.

    Returns: ||delta_W||_F / ||U_r^T W V_r||_F
    """
    W = np.asarray(W, dtype=float)
    delta_W = np.asarray(delta_W, dtype=float)

    # SVD of the update: delta_W = U S V^T
    U, S, Vt = np.linalg.svd(delta_W, full_matrices=False)

    # Top-r left and right singular subspaces
    U_r = U[:, :r]          # d x r
    V_r = Vt[:r, :].T       # k x r  (transpose of the top r rows of Vt)

    # Project W into the pair of subspaces: U_r^T W V_r  (r x r)
    proj = U_r.T @ W @ V_r

    norm_delta = np.linalg.norm(delta_W, 'fro')
    norm_proj = np.linalg.norm(proj, 'fro')

    if norm_proj == 0.0:
        return float('inf')

    return float(norm_delta / norm_proj)