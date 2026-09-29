import numpy as np

def compress_kv(K, w, b, m):
    """
    Compress a key sequence by aggregating every m consecutive entries into one
    using a content-dependent softmax with a positional bias.
    """
    K = np.asarray(K, dtype=float)
    w = np.asarray(w, dtype=float)
    b = np.asarray(b, dtype=float)

    S, D = K.shape
    num_blocks = S // m

    # Reshape into (num_blocks, m, D)
    K_blocks = K.reshape(num_blocks, m, D)

    # Content scores: w · K_{j*m+i} for each block position -> (num_blocks, m)
    content_scores = K_blocks @ w  # (num_blocks, m)

    # Add positional bias (broadcast over blocks)
    scores = content_scores + b  # (num_blocks, m)

    # Softmax over the m positions within each block
    # Subtract max for numerical stability
    scores = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    alpha = exp_scores / exp_scores.sum(axis=1, keepdims=True)  # (num_blocks, m)

    # Weighted sum of block keys: (num_blocks, m, 1) * (num_blocks, m, D)
    compressed = (alpha[:, :, None] * K_blocks).sum(axis=1)  # (num_blocks, D)

    return compressed.tolist()