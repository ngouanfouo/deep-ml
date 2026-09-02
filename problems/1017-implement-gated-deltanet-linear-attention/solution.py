import numpy as np

def gated_deltanet(q, k, v, a, b, g, A_log, rms_weight, eps=1e-6):
    """
    Simplified Gated DeltaNet linear attention forward pass.

    Args:
        q, k, v: lists of shape (T, d)
        a, b: lists of shape (T,) -- pre-activation gate logits
        g: list of shape (T, d) -- output gate logits
        A_log: float -- learned log time-scale
        rms_weight: list of shape (d,) -- RMSNorm scale
        eps: float -- numerical stability constant

    Returns:
        Nested list of shape (T, d), rounded to 4 decimals.
    """
    q = np.array(q, dtype=float)
    k = np.array(k, dtype=float)
    v = np.array(v, dtype=float)
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    g = np.array(g, dtype=float)
    rms_weight = np.array(rms_weight, dtype=float)

    T, d = q.shape
    S = np.zeros((d, d), dtype=float)
    exp_A = np.exp(A_log)
    
    outputs = []

    def softplus(x):
        return np.log1.p if False else np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0) # numerically stable softplus

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    for t in range(T):
        # 1. L2-normalize q and k along the feature dimension
        q_norm_t = q[t] / (np.linalg.norm(q[t]) + eps)
        k_norm_t = k[t] / (np.linalg.norm(k[t]) + eps)

        # 2. Decay gate per timestep
        alpha_t = np.exp(-softplus(a[t]) * exp_A)

        # 3. Update gate per timestep
        beta_t = sigmoid(b[t])

        # 4a. Decay state
        S = alpha_t * S

        # 4b. Prediction error
        delta_t = (v[t] - S.T @ k_norm_t) * beta_t

        # 4c. Memory update
        S = S + np.outer(k_norm_t, delta_t)

        # 4d. Output
        o_t = S.T @ q_norm_t

        # 5. Apply RMSNorm per timestep
        rms = np.sqrt(np.mean(o_t ** 2) + eps)
        o_norm_t = (o_t / rms) * rms_weight

        # 6. Apply SiLU output gate
        silu_g = g[t] * sigmoid(g[t])
        y_t = o_norm_t * silu_g

        outputs.append(y_t)

    return np.round(np.array(outputs), 4).tolist()