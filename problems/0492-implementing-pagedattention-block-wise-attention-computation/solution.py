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
    
    # 1. Reconstruct the contiguous key and value tensors up to context_len
    # Allocating arrays to collect valid tokens for this sequence
    K_list = []
    V_list = []
    
    tokens_gathered = 0
    for logical_idx, physical_block_idx in enumerate(block_table):
        if tokens_gathered >= context_len:
            break
            
        # Determine how many tokens inside this block are valid
        tokens_in_block = min(block_size, context_len - tokens_gathered)
        
        # Extract the valid slices: shape -> (tokens_in_block, num_heads, head_dim)
        k_block_slice = key_cache[physical_block_idx, :tokens_in_block]
        v_block_slice = value_cache[physical_block_idx, :tokens_in_block]
        
        K_list.append(k_block_slice)
        V_list.append(v_block_slice)
        
        tokens_gathered += tokens_in_block
        
    # Concatenate along the token dimension: shape -> (context_len, num_heads, head_dim)
    K = np.concatenate(K_list, axis=0)
    V = np.concatenate(V_list, axis=0)
    
    # 2. Reshape/transpose for head-independent attention calculation
    # Query: (num_heads, head_dim) -> (num_heads, 1, head_dim)
    Q = query[:, np.newaxis, :] 
    
    # Keys: (context_len, num_heads, head_dim) -> (num_heads, context_len, head_dim)
    K = np.transpose(K, (1, 0, 2))
    
    # Values: (context_len, num_heads, head_dim) -> (num_heads, context_len, head_dim)
    V = np.transpose(V, (1, 0, 2))
    
    # 3. Scaled dot-product attention
    # Attention scores shape: (num_heads, 1, context_len)
    # Equivalent to multiplying Q with K transposed on its last two axes
    scores = np.matmul(Q, np.transpose(K, (0, 2, 1))) / np.sqrt(head_dim)
    
    # Numerically stable Softmax over the context_len dimension
    scores_max = np.max(scores, axis=-1, keepdims=True)
    attn_weights = np.exp(scores - scores_max)
    attn_weights /= np.sum(attn_weights, axis=-1, keepdims=True)
    
    # Calculate output: (num_heads, 1, context_len) x (num_heads, context_len, head_dim)
    # Output shape: (num_heads, 1, head_dim)
    context_out = np.matmul(attn_weights, V)
    
    # Squeeze out the dummy sequence length dimension to get (num_heads, head_dim)
    output = np.squeeze(context_out, axis=1)
    
    # 4. Round to 4 decimal places
    return np.round(output, 4)