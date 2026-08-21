import numpy as np

def deterministic_sparse_attention_backward(contributions: list, num_sms: int, kv_len: int, dim: int) -> list:
    """
    Deterministically accumulate sparse attention KV gradient contributions.

    Args:
        contributions: list of dicts with keys 'sm_id' (int), 'kv_idx' (int),
                       'grad' (list[float] of length dim)
        num_sms: number of SMs (size of per-SM buffer axis)
        kv_len: KV sequence length
        dim: feature dimension

    Returns:
        Nested list of shape (kv_len, dim) containing the reduced gradient.
    """
    # Initialize per-SM buffers: shape (num_sms, kv_len, dim)
    sm_buffers = np.zeros((num_sms, kv_len, dim), dtype=np.float64)
    
    # Accumulate contributions into each SM's private buffer
    for contribution in contributions:
        sm_id = contribution['sm_id']
        kv_idx = contribution['kv_idx']
        grad = contribution['grad']
        
        # Add gradient to the appropriate SM's buffer
        sm_buffers[sm_id, kv_idx, :] += np.array(grad, dtype=np.float64)
    
    # Reduce per-SM buffers into final output
    # Iterate SM ids in ascending order (0, 1, 2, ...)
    final_grad = np.zeros((kv_len, dim), dtype=np.float64)
    for sm_id in range(num_sms):
        final_grad += sm_buffers[sm_id]
    
    # Convert to nested list
    return final_grad.tolist()