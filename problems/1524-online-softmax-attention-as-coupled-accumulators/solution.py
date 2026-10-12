import math
import numpy as np


def attend_step(m, l, o, s, v):
    m_new = max(m, s)
    alpha = math.exp(m - m_new) if m != -math.inf else 0.0
    p = math.exp(s - m_new)
    l_new = l * alpha + p
    o_new = o * alpha + p * v
    return m_new, l_new, o_new


def flash_attention(Q, K, V):
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    N, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    out = np.zeros((N, d), dtype=np.float32)
    for i in range(N):
        q = Q[i]
        m = -math.inf
        l = 0.0
        o = np.zeros(d, dtype=np.float64)
        for j in range(K.shape[0]):
            s = float(np.dot(q, K[j]) * scale)
            m, l, o = attend_step(m, l, o, s, V[j])
        out[i] = (o / l).astype(np.float32)
    return out


def reference_attention(Q, K, V):
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    d = Q.shape[1]
    scores = Q @ K.T / math.sqrt(d)
    scores = scores - scores.max(axis=1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / weights.sum(axis=1, keepdims=True)
    return (weights @ V).astype(np.float32)


def score_matrix_bytes(N, elem_bytes=4):
    return N * N * elem_bytes