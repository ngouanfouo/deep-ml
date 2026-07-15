import numpy as np

def ring_attention_simulate(Q: np.ndarray, K: np.ndarray, V: np.ndarray, num_devices: int) -> tuple:
    """
    Simulate Ring Attention for Context Parallelism.
    """
    seq_len, d = Q.shape
    C = seq_len // num_devices  # Chunk size per device
    
    # Split Q, K, V into chunks
    Q_chunks = [Q[i*C:(i+1)*C] for i in range(num_devices)]
    K_chunks = [K[i*C:(i+1)*C] for i in range(num_devices)]
    V_chunks = [V[i*C:(i+1)*C] for i in range(num_devices)]
    
    # Initialize per-device online softmax state
    # Each device maintains: max_score, sum_exp, output_accum
    max_scores = [np.full((C, 1), -np.inf) for _ in range(num_devices)]
    sum_exps = [np.zeros((C, 1)) for _ in range(num_devices)]
    output_accums = [np.zeros((C, d)) for _ in range(num_devices)]
    
    # Communication schedule: source device index for each device at each step
    comm_schedule = []
    
    # Ring steps: each device processes KV from a different source each step
    for step in range(num_devices):
        # Determine which KV chunk each device processes at this step
        # Device i processes KV from source (i - step) % num_devices
        # This rotates KV chunks in the ring
        source_indices = [(i - step) % num_devices for i in range(num_devices)]
        comm_schedule.append(source_indices)
        
        # Each device computes attention with its Q and the current KV chunk
        for device_idx in range(num_devices):
            Q_local = Q_chunks[device_idx]
            kv_source = source_indices[device_idx]
            K_local = K_chunks[kv_source]
            V_local = V_chunks[kv_source]
            
            # Compute scaled dot-product attention scores
            # scores: (C, C) where C is chunk size
            scores = (Q_local @ K_local.T) / np.sqrt(d)
            
            # Online softmax update
            # Find new max
            new_max = np.maximum(max_scores[device_idx], np.max(scores, axis=1, keepdims=True))
            
            # Compute exp(scores - new_max) and exp(old_max - new_max)
            exp_scores = np.exp(scores - new_max)
            exp_old = np.exp(max_scores[device_idx] - new_max)
            
            # Update sum of exps
            new_sum = sum_exps[device_idx] * exp_old + np.sum(exp_scores, axis=1, keepdims=True)
            
            # Update output accumulation
            # V_local is (C, d), exp_scores is (C, C)
            # exp_scores @ V_local gives (C, d)
            new_output = output_accums[device_idx] * exp_old + (exp_scores @ V_local)
            
            # Update state
            max_scores[device_idx] = new_max
            sum_exps[device_idx] = new_sum
            output_accums[device_idx] = new_output
    
    # Normalize outputs for each device
    normalized_outputs = []
    for device_idx in range(num_devices):
        # output = output_accum / sum_exp
        normalized = output_accums[device_idx] / sum_exps[device_idx]
        normalized_outputs.append(normalized)
    
    # Concatenate outputs in order
    output = np.vstack(normalized_outputs)
    
    # Round to 4 decimal places
    output = np.round(output, 4)
    
    return output, comm_schedule