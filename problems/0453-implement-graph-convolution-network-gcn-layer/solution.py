import numpy as np


def gcn_layer(A: np.ndarray, X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Perform a single GCN layer forward pass.

    Args:
        A: Adjacency matrix of shape (N, N)
        X: Node feature matrix of shape (N, F_in)
        W: Weight matrix of shape (F_in, F_out)

    Returns:
        Output feature matrix of shape (N, F_out)
    """
    N = A.shape[0]

    # 1) Self-loop augmentation: A_tilde = A + I
    A_tilde = A + np.eye(N)

    # 2) Symmetric degree normalization: D^(-1/2) @ A_tilde @ D^(-1/2)
    deg = A_tilde.sum(axis=1)                          # (N,)
    # Avoid division by zero for isolated nodes (degenerate case)
    deg_inv_sqrt = np.zeros_like(deg, dtype=float)
    nonzero = deg > 0
    deg_inv_sqrt[nonzero] = 1.0 / np.sqrt(deg[nonzero])

    D_inv_sqrt = np.diag(deg_inv_sqrt)
    A_norm = D_inv_sqrt @ A_tilde @ D_inv_sqrt         # (N, N)

    # 3) Neighborhood aggregation + linear transform
    aggregated = A_norm @ X                            # (N, F_in)
    Z = aggregated @ W                                 # (N, F_out)

    # 4) ReLU activation
    return np.maximum(Z, 0.0)