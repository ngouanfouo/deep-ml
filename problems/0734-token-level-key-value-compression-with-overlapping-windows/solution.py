import numpy as np

def compressed_kv_entries(H, W_aKV, W_bKV, W_aZ, W_bZ, B_a, B_b, m, stride):
    """
    Compresses a sequence of key-value tokens by combining windows of size m 
    across two parallel streams using normalized compression weights.

    Args:
        H: Hidden states of shape (n, d)
        W_aKV, W_bKV: KV projection weights for streams a and b, shape (d, c)
        W_aZ, W_bZ: Compression-weight projection weights, shape (d, c)
        B_a, B_b: Positional biases for streams a and b, shape (m, c)
        m: Window size (number of consecutive tokens per group)
        stride: Step size between consecutive windows

    Returns:
        List of lists of shape ((n - m) // stride + 1, c) containing compressed KV entries.
    """
    H = np.asarray(H, dtype=float)
    W_aKV = np.asarray(W_aKV, dtype=float)
    W_bKV = np.asarray(W_bKV, dtype=float)
    W_aZ = np.asarray(W_aZ, dtype=float)
    W_bZ = np.asarray(W_bZ, dtype=float)
    B_a = np.asarray(B_a, dtype=float)
    B_b = np.asarray(B_b, dtype=float)

    n, d = H.shape
    num_groups = (n - m) // stride + 1

    if num_groups <= 0:
        return []

    # Linear projections for KV values and compression logits for both streams
    KV_a = H @ W_aKV  # (n, c)
    KV_b = H @ W_bKV  # (n, c)
    Z_a = H @ W_aZ    # (n, c)
    Z_b = H @ W_bZ    # (n, c)

    compressed_entries = []

    for g in range(num_groups):
        start = g * stride
        end = start + m

        # Extract window tokens and add positional biases
        kv_a_win = KV_a[start:end]
        kv_b_win = KV_b[start:end]
        z_a_win = Z_a[start:end] + B_a
        z_b_win = Z_b[start:end] + B_b

        # Concatenate tokens from both streams (2m total tokens in the window)
        kv_concat = np.concatenate([kv_a_win, kv_b_win], axis=0)  # (2m, c)
        z_concat = np.concatenate([z_a_win, z_b_win], axis=0)    # (2m, c)

        # Softmax normalization across the 2m tokens per channel dimension
        z_max = np.max(z_concat, axis=0, keepdims=True)
        exp_z = np.exp(z_concat - z_max)
        weights = exp_z / np.sum(exp_z, axis=0, keepdims=True)  # (2m, c)

        # Compute weighted sum of KV entries
        compressed_kv = np.sum(weights * kv_concat, axis=0)      # (c,)
        compressed_entries.append(compressed_kv)

    # Convert to Python list rounded appropriately or standard list format
    return np.array(compressed_entries).tolist()