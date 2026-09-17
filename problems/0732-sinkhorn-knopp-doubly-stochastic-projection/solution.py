import numpy as np

def sinkhorn_knopp(B: list, t_max: int = 20) -> list:
    """
    Project a square matrix onto the set of doubly stochastic matrices.

    Args:
        B: n x n matrix as a list of lists (real-valued).
        t_max: number of normalization iterations.

    Returns:
        A nested list representing the resulting doubly stochastic matrix.
    """
    M = np.exp(np.asarray(B, dtype=float))

    for _ in range(t_max):
        # Column normalization: divide each column by its sum
        col_sums = M.sum(axis=0, keepdims=True)
        M = M / col_sums

        # Row normalization: divide each row by its sum
        row_sums = M.sum(axis=1, keepdims=True)
        M = M / row_sums

    return M.tolist()