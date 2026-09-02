import numpy as np

def batchnorm1d_backward_fused(x: np.ndarray, bngain: np.ndarray, dhpreact: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """
    Compute the fused backward pass through a BatchNorm1d layer.

    Args:
        x:        pre-BN input, shape (N, D)
        bngain:   scale parameter, shape (1, D)
        dhpreact: upstream gradient, shape (N, D)
        eps:      numerical stability constant

    Returns:
        dhprebn: gradient w.r.t. x, shape (N, D)
    """
    N, D = x.shape

    # Forward stats
    mean = x.mean(axis=0, keepdims=True)                    # (1, D)
    var = x.var(axis=0, ddof=1, keepdims=True)              # (1, D)
    std_inv = 1.0 / np.sqrt(var + eps)                      # (1, D)
    x_hat = (x - mean) * std_inv                            # (N, D)

    # Sums over batch
    sum_dh = dhpreact.sum(axis=0, keepdims=True)            # (1, D)
    sum_dh_xhat = (dhpreact * x_hat).sum(axis=0, keepdims=True)  # (1, D)

    # Fused gradient
    inside = (N * dhpreact
              - sum_dh
              - (N / (N - 1)) * x_hat * sum_dh_xhat)       # (N, D)

    dhprebn = (bngain * std_inv / N) * inside              # (N, D)

    return dhprebn