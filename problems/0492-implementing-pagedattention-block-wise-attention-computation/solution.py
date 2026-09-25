import numpy as np

def paged_attention(
    query: np.ndarray,
    key_cache: np.ndarray,
    value_cache: np.ndarray,
    block_table: list,
    context_len: int
) -> np.ndarray:
    """
    Perform scaled dot-product attention with paged KV cache.

    Args:
        query:       (num_heads, head_dim) query for a single token
        key_cache:   (num_physical_blocks, block_size, num_heads, head_dim)
        value_cache: (num_physical_blocks, block_size, num_heads, head_dim)
        block_table: list of physical block indices (logical -> physical)
        context_len: number of valid KV tokens in the sequence

    Returns:
        (num_heads, head_dim) attention output, rounded to 4 decimal places
    """
    num_heads, head_dim = query.shape
    block_size = key_cache.shape[1]

    # 1. Reconstruct contiguous K and V by walking the page table in
    #    logical order, stopping at context_len tokens.
    K_list = []
    V_list = []

    tokens_gathered = 0
    for physical_block_idx in block_table:
        if tokens_gathered >= context_len:
            break

        # The last logical block may be only partially filled.
        tokens_in_block = min(block_size, context_len - tokens_gathered)

        K_list.append(key_cache[physical_block_idx, :tokens_in_block])
        V_list.append(value_cache[physical_block_idx, :tokens_in_block])

        tokens_gathered += tokens_in_block

    # (context_len, num_heads, head_dim)
    K = np.concatenate(K_list, axis=0)
    V = np.concatenate(V_list, axis=0)

    # 2. Rearrange to put heads first for per-head attention.
    # Q: (num_heads, 1, head_dim)
    Q = query[:, np.newaxis, :]
    # K, V: (num_heads, context_len, head_dim)
    K = np.transpose(K, (1, 0, 2))
    V = np.transpose(V, (1, 0, 2))

    # 3. Scaled dot-product attention with numerically stable softmax.
    # scores: (num_heads, 1, context_len)
    scores = np.matmul(Q, np.transpose(K, (0, 2, 1))) / np.sqrt(head_dim)

    scores_max = np.max(scores, axis=-1, keepdims=True)
    attn_weights = np.exp(scores - scores_max)
    attn_weights /= np.sum(attn_weights, axis=-1, keepdims=True)

    # (num_heads, 1, head_dim)
    context_out = np.matmul(attn_weights, V)

    # Squeeze the dummy sequence-length dimension.
    output = np.squeeze(context_out, axis=1)

    return np.round(output, 4)