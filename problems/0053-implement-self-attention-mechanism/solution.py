import numpy as np


def compute_qkv(X, W_q, W_k, W_v):
    """Compute Query, Key, Value matrices from input X and weight matrices."""
    Q = np.dot(X, W_q)
    K = np.dot(X, W_k)
    V = np.dot(X, W_v)
    return Q, K, V


def self_attention(Q, K, V):
    """Compute scaled dot-product self-attention.

    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_v)

    Returns:
        Attention output of shape (seq_len, d_v)
    """
    # 1. Get the dimensionality of the key vectors (d_k)
    d_k = Q.shape[1]

    # 2. Compute attention scores: Q * K^T / sqrt(d_k)
    scores = np.dot(Q, K.T) / np.sqrt(d_k)

    # 3. Apply stable row-wise softmax to obtain attention weights
    # Subtracting the row max prevents exponential overflow
    shifted_scores = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(shifted_scores)
    attention_weights = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # 4. Compute the final contextualized output: attention_weights * V
    output = np.dot(attention_weights, V)

    return output