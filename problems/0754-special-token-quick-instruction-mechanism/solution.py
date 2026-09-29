import numpy as np

def special_token_attention(Q, K, V, num_base, num_special):
    """
    Q, K, V: array-like of shape (num_base + num_special, d)
    num_base: int, number of base tokens (L)
    num_special: int, number of appended special tokens (M)
    Returns: list of M lists (size d), attention outputs for the special tokens,
    rounded to 4 decimal places.
    """
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)

    L = num_base
    M = num_special
    d = Q.shape[1]
    scale = 1.0 / np.sqrt(d)

    outputs = []
    for i in range(M):
        special_pos = L + i
        # Allowed positions: all base tokens + the special token itself
        allowed = list(range(L)) + [special_pos]

        q = Q[special_pos]                      # (d,)
        k_allowed = K[allowed]                  # (len(allowed), d)
        v_allowed = V[allowed]                  # (len(allowed), d)

        scores = (k_allowed @ q) * scale        # (len(allowed),)

        # Numerically stable softmax
        scores = scores - np.max(scores)
        exp_scores = np.exp(scores)
        alpha = exp_scores / np.sum(exp_scores)

        out = alpha @ v_allowed                 # (d,)
        outputs.append([round(float(x), 4) for x in out])

    return outputs