import numpy as np


def multi_head_cross_attention(decoder_x, encoder_out, W_q, W_k, W_v, W_o, num_heads, src_pad_mask=None):
    """
    Batched multi-head encoder-decoder attention.

    decoder_x:    (B, T, d_model)  queries come from here
    encoder_out:  (B, S, d_model)  keys and values come from here
    W_q, W_k, W_v, W_o: (d_model, d_model)
    num_heads:    H, with d_model divisible by H
    src_pad_mask: optional (B, S) boolean / 0-1 array, True or 1 at padded source positions

    Returns (output, weights) with shapes (B, T, d_model) and (B, H, T, S).
    Masked source positions must have weight exactly 0 for every head and target position.
    """
    decoder_x = np.asarray(decoder_x, dtype=float)
    encoder_out = np.asarray(encoder_out, dtype=float)
    W_q = np.asarray(W_q, dtype=float)
    W_k = np.asarray(W_k, dtype=float)
    W_v = np.asarray(W_v, dtype=float)
    W_o = np.asarray(W_o, dtype=float)

    B, T, d_model = decoder_x.shape
    S = encoder_out.shape[1]
    H = num_heads
    d_k = d_model // H

    # Linear projections (queries from decoder, keys/values from encoder).
    Q = decoder_x @ W_q          # (B, T, d_model)
    K = encoder_out @ W_k        # (B, S, d_model)
    V = encoder_out @ W_v        # (B, S, d_model)

    # Split into heads: (B, T, H, d_k) -> (B, H, T, d_k).
    Q = Q.reshape(B, T, H, d_k).transpose(0, 2, 1, 3)  # (B, H, T, d_k)
    K = K.reshape(B, S, H, d_k).transpose(0, 2, 1, 3)  # (B, H, S, d_k)
    V = V.reshape(B, S, H, d_k).transpose(0, 2, 1, 3)  # (B, H, S, d_k)

    # Scaled dot-product scores per head.
    scores = (Q @ K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)  # (B, H, T, S)

    # Mask padded source positions with a very negative value so softmax gives exactly 0.
    if src_pad_mask is not None:
        pad = np.asarray(src_pad_mask, dtype=bool)  # (B, S)
        scores = np.where(pad[:, None, None, :], -1e9, scores)

    # Softmax over the source axis; each row still sums to 1 (>=1 real token per row).
    m = np.max(scores, axis=-1, keepdims=True)
    e = np.exp(scores - m)
    weights = e / np.sum(e, axis=-1, keepdims=True)  # (B, H, T, S)

    # Weighted values, concatenate heads, project with W_o.
    out = weights @ V                                   # (B, H, T, d_k)
    out = out.transpose(0, 2, 1, 3).reshape(B, T, d_model)  # (B, T, d_model)
    out = out @ W_o                                     # (B, T, d_model)

    return out, weights