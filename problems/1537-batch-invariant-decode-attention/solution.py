import numpy as np


def _block_partial(q, Kb, Vb, scale):
    """Online-softmax partial (m, l, a) for one block of keys/values."""
    s = (Kb @ q) * scale          # (len,)
    m = np.max(s)
    e = np.exp(s - m)             # (len,)
    l = np.sum(e)                 # scalar
    a = e @ Vb                    # (d,)
    return m, l, a


def _merge(p0, p1):
    """Merge two partials exactly as the spec's online-softmax rule."""
    m0, l0, a0 = p0
    m1, l1, a1 = p1
    m = m0 if m0 >= m1 else m1
    c0 = np.exp(m0 - m)
    c1 = np.exp(m1 - m)
    l = l0 * c0 + l1 * c1
    a = a0 * c0 + a1 * c1
    return m, l, a


def attention_single_pass(q, K, V, block):
    """
    Whole-sequence decode attention on one 'SM': walk the KV cache in blocks of `block` rows,
    form each block's online-softmax partial (max, sum, acc) and merge it into the running
    state in order. q: (d,), K, V: (n, d). Returns acc / sum with shape (d,).
    """
    q = np.asarray(q)
    K = np.asarray(K)
    V = np.asarray(V)

    n = K.shape[0]
    d = q.shape[0]
    scale = 1.0 / float(np.sqrt(d))

    # First block's partial.
    end = min(block, n)
    running = _block_partial(q, K[0:end], V[0:end], scale)

    # Merge each later block in order.
    start = end
    while start < n:
        end = min(start + block, n)
        p = _block_partial(q, K[start:end], V[start:end], scale)
        running = _merge(running, p)
        start = end

    _, l, a = running
    return a / l


def attention_split_kv(q, K, V, block, num_splits):
    """
    Split-KV decode attention: the blocks are divided into num_splits contiguous groups of
    ceil(num_blocks / num_splits) blocks, each group computes its block partials
    independently, then ALL block partials are merged in global block order with the same
    merge rule as attention_single_pass. Must be bitwise equal to attention_single_pass.
    """
    q = np.asarray(q)
    K = np.asarray(K)
    V = np.asarray(V)

    n = K.shape[0]
    d = q.shape[0]
    scale = 1.0 / float(np.sqrt(d))

    num_blocks = (n + block - 1) // block
    group_size = (num_blocks + num_splits - 1) // num_splits if num_splits > 0 else num_blocks

    # Compute every block's partial in global block order (groups don't change the partials).
    partials = []
    for b in range(num_blocks):
        start = b * block
        end = min(start + block, n)
        partials.append(_block_partial(q, K[start:end], V[start:end], scale))

    # Merge all block partials in global block order, starting from the first.
    running = partials[0]
    for p in partials[1:]:
        running = _merge(running, p)

    _, l, a = running
    return a / l


def batch_invariant_attention(Q, K, V, block, num_sms):
    """
    Q: (B, d), K, V: (B, n, d). If B >= num_sms use the single-pass kernel for every sequence,
    otherwise the split kernel with num_splits = max(1, min(num_blocks, num_sms // B)).
    Returns (out, kernel) with out of shape (B, d) in K's dtype and kernel in {"single", "split"}.
    """
    Q = np.asarray(Q)
    K = np.asarray(K)
    V = np.asarray(V)

    B, n, d = K.shape
    out = np.empty((B, d), dtype=K.dtype)

    if B >= num_sms:
        kernel = "single"
        for b in range(B):
            out[b] = attention_single_pass(Q[b], K[b], V[b], block)
    else:
        kernel = "split"
        num_blocks = (n + block - 1) // block
        num_splits = max(1, min(num_blocks, num_sms // B))
        for b in range(B):
            out[b] = attention_split_kv(Q[b], K[b], V[b], block, num_splits)

    return out, kernel