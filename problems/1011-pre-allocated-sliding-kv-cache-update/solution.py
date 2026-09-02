import numpy as np

def sliding_kv_cache_update(cache_k, cache_v, current_len, new_k, new_v, window_size):
    """
    Update a pre-allocated sliding-window KV cache and produce the causal mask
    for the new queries against the updated cache.

    Args:
        cache_k, cache_v: nested lists / arrays of shape
            (batch, num_heads, window_size, head_dim).
        current_len: int, number of valid entries currently in the cache (<= window_size).
        new_k, new_v: nested lists / arrays of shape
            (batch, num_heads, new_len, head_dim).
        window_size: int, maximum cache length.

    Returns:
        dict with keys 'cache_k', 'cache_v', 'current_len', 'mask'.
    """
    # Create copies to avoid mutating original buffers directly
    Ck = np.array(cache_k, dtype=float).copy()
    Cv = np.array(cache_v, dtype=float).copy()
    Nk = np.array(new_k, dtype=float)
    Nv = np.array(new_v, dtype=float)
    
    new_len = Nk.shape[2]
    total_len = current_len + new_len
    updated_len = int(min(total_len, window_size))
    
    # Extract currently valid old entries and concatenate with new entries
    combined_k = np.concatenate([Ck[:, :, :current_len, :], Nk], axis=2)
    combined_v = np.concatenate([Cv[:, :, :current_len, :], Nv], axis=2)
    
    # Keep only the most recent entries up to window_size
    kept_k = combined_k[:, :, -updated_len:, :]
    kept_v = combined_v[:, :, -updated_len:, :]
    
    # Write back into the cache buffers from index 0 to updated_len - 1
    Ck[:, :, :updated_len, :] = kept_k
    Cv[:, :, :updated_len, :] = kept_v
    
    # Generate causal mask, clamping query positions to 0 if they fall outside the window
    query_positions = np.maximum(0, updated_len - new_len + np.arange(new_len))[:, None]
    cache_positions = np.arange(updated_len)[None, :]
    mask = (cache_positions <= query_positions).astype(int).tolist()
    
    return {
        'cache_k': Ck.tolist(),
        'cache_v': Cv.tolist(),
        'current_len': updated_len,
        'mask': mask
    }