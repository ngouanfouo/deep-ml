import numpy as np

def mhc_forward(
    x: np.ndarray,
    H_pre_raw: np.ndarray,
    H_post_raw: np.ndarray,
    H_res_raw: np.ndarray,
    layer_output: np.ndarray,
    sinkhorn_iters: int = 5
) -> np.ndarray:
    """
    Compute mHC forward pass with manifold-constrained mappings.
    
    The mHC formula is:
        x_out = H_res @ x + H_post.T @ layer_output
    
    Where:
        - H_pre (not used here, already applied to get layer_output)
        - H_post = 2 * sigmoid(H_post_raw)  (non-negative constraint)
        - H_res = Sinkhorn(exp(H_res_raw))  (doubly stochastic constraint)
    
    Args:
        x: Hidden states, shape (n, C) where n is num streams
        H_pre_raw: Raw pre-mapping coefficients, shape (1, n) - not used in this step
        H_post_raw: Raw post-mapping coefficients, shape (1, n)
        H_res_raw: Raw residual mapping coefficients, shape (n, n)
        layer_output: Output from layer F, shape (1, C)
        sinkhorn_iters: Number of Sinkhorn iterations
    
    Returns:
        x_out: Updated hidden states, shape (n, C)
    
    Notes:
        - sigmoid(z) = 1 / (1 + exp(-z))
        - Sinkhorn: start with exp(H_res_raw), alternate row/column normalization
    """
    # Input validation
    if x.ndim != 2:
        raise ValueError(f"x must be 2D, got shape {x.shape}")
    
    if H_res_raw.shape[0] != H_res_raw.shape[1]:
        raise ValueError(f"H_res_raw must be square, got shape {H_res_raw.shape}")
    
    n_streams = x.shape[0]
    if H_res_raw.shape[0] != n_streams:
        raise ValueError(f"H_res_raw must have shape ({n_streams}, {n_streams}), got {H_res_raw.shape}")
    
    if H_post_raw.shape[1] != n_streams:
        raise ValueError(f"H_post_raw must have shape (1, {n_streams}), got {H_post_raw.shape}")
    
    # Step 1: Compute H_post from raw coefficients
    # H_post = 2 * sigmoid(H_post_raw)
    # Shape: (1, n)
    # Use numerical stable sigmoid: clip values to avoid overflow
    H_post_raw_clipped = np.clip(H_post_raw, -500, 500)
    H_post = 2 * (1 / (1 + np.exp(-H_post_raw_clipped)))
    
    # Step 2: Compute H_res via Sinkhorn iteration
    # Start with element-wise exponential of H_res_raw
    # Clip to avoid overflow
    H_res_raw_clipped = np.clip(H_res_raw, -500, 500)
    H_res = np.exp(H_res_raw_clipped)
    
    # Sinkhorn iteration: alternate row and column normalization
    # This ensures the matrix is doubly stochastic (rows and columns sum to 1)
    for i in range(sinkhorn_iters):
        # Row normalization: divide each row by its sum
        row_sums = H_res.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums = np.maximum(row_sums, 1e-12)
        H_res = H_res / row_sums
        
        # Column normalization: divide each column by its sum
        col_sums = H_res.sum(axis=0, keepdims=True)
        # Avoid division by zero
        col_sums = np.maximum(col_sums, 1e-12)
        H_res = H_res / col_sums
    
    # Step 3: Compute the mHC output
    # x_out = H_res @ x + H_post.T @ layer_output
    # H_res @ x: (n, n) @ (n, C) -> (n, C)
    # H_post.T @ layer_output: (n, 1) @ (1, C) -> (n, C)
    residual_contribution = H_res @ x
    post_contribution = H_post.T @ layer_output
    x_out = residual_contribution + post_contribution
    
    return x_out