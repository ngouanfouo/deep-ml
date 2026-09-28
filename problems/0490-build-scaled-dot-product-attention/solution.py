import numpy as np

def scaled_dot_product_attention(Q: np.ndarray,
                                 K: np.ndarray,
                                 V: np.ndarray,
                                 mask: np.ndarray = None) -> tuple:
    """
    Compute Scaled Dot-Product Attention.

    Args:
        Q: Query matrix of shape (seq_len_q, d_k)
        K: Key matrix of shape (seq_len_k, d_k)
        V: Value matrix of shape (seq_len_k, d_v)
        mask: Optional binary mask of shape (seq_len_q, seq_len_k)
              1 = keep, 0 = block

    Returns:
        Tuple of (output, attention_weights)
    """
    d_k = Q.shape[-1]

    # 1) Scaled scores
    scores = Q @ K.T / np.sqrt(d_k)

    # 2) Apply mask: blocked positions become -inf before softmax
    if mask is not None:
        mask = np.asarray(mask)
        scores = np.where(mask == 1, scores, -np.inf)

    # 3) Numerically stable softmax
    row_max = np.max(scores, axis=-1, keepdims=True)

    # For rows that are fully masked, avoid -inf - -inf = NaN
    safe_row_max = np.where(np.isfinite(row_max), row_max, 0.0)
    shifted = scores - safe_row_max
    exp_scores = np.exp(shifted)

    # Normalize; rows with zero total (fully masked) yield zero weights
    sum_exp = np.sum(exp_scores, axis=-1, keepdims=True)
    attention_weights = np.divide(
        exp_scores,
        sum_exp,
        out=np.zeros_like(exp_scores),
        where=sum_exp > 0
    )

    # 4) Weighted sum of values
    output = attention_weights @ V

    return output, attention_weights