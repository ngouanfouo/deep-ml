import numpy as np

def manual_cross_entropy_backward(logits: np.ndarray, y: np.ndarray) -> dict:
    """
    Manually compute gradients through every intermediate of softmax + NLL loss.

    Args:
        logits: numpy array of shape (n, C)
        y:      numpy array of integer class indices, shape (n,)

    Returns:
        Dictionary with keys 'dlogprobs', 'dprobs', 'dcounts_sum_inv',
        'dcounts_sum', 'dcounts', 'dnorm_logits', 'dlogit_maxes', 'dlogits'.
    """
    # Forward pass (compute intermediates)
    logit_maxes = logits.max(axis=1, keepdims=True)          # (n, 1)
    norm_logits = logits - logit_maxes                      # (n, C)
    counts = np.exp(norm_logits)                            # (n, C)
    counts_sum = counts.sum(axis=1, keepdims=True)          # (n, 1)
    counts_sum_inv = counts_sum ** -1                      # (n, 1)
    probs = counts * counts_sum_inv                        # (n, C)
    logprobs = np.log(probs)                               # (n, C)

    n, C = logits.shape

    # Seed gradient: dloss / dlogprobs
    dlogprobs = np.zeros_like(logprobs)
    dlogprobs[np.arange(n), y] = -1.0 / n                  # (n, C)

    # Backward through logprobs = log(probs)
    dprobs = dlogprobs / probs                             # (n, C)

    # Backward through probs = counts * counts_sum_inv
    dcounts_sum_inv = np.sum(dprobs * counts, axis=1, keepdims=True)  # (n, 1)
    dcounts_sum = -dcounts_sum_inv / (counts_sum ** 2)                # (n, 1)

    # counts receives two contributions:
    # 1) from probs: dprobs * counts_sum_inv
    # 2) from counts_sum: dcounts_sum (broadcast to all columns)
    dcounts = dprobs * counts_sum_inv + dcounts_sum                  # (n, C)

    # Backward through counts = exp(norm_logits)
    dnorm_logits = dcounts * counts                                 # (n, C)

    # Backward through norm_logits = logits - logit_maxes
    dlogit_maxes = -np.sum(dnorm_logits, axis=1, keepdims=True)     # (n, 1)

    # Backward through logit_maxes = max(logits)
    # one‑hot at the argmax (first occurrence on ties)
    max_indices = np.argmax(logits, axis=1)                         # (n,)
    one_hot_max = np.zeros_like(logits)
    one_hot_max[np.arange(n), max_indices] = 1.0

    dlogits = dnorm_logits + dlogit_maxes * one_hot_max            # (n, C)

    return {
        'dlogprobs': dlogprobs,
        'dprobs': dprobs,
        'dcounts_sum_inv': dcounts_sum_inv,
        'dcounts_sum': dcounts_sum,
        'dcounts': dcounts,
        'dnorm_logits': dnorm_logits,
        'dlogit_maxes': dlogit_maxes,
        'dlogits': dlogits,
    }