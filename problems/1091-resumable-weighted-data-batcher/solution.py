import numpy as np


def weighted_batcher(datasets, weights, num_samples, start_offset=0):
    """
    Emits items by drawing from several datasets according to integer weights,
    supporting deterministic resumption via start_offset.
    """
    if start_offset >= num_samples:
        return []
        
    K = len(datasets)
    weights = list(weights)
    P = sum(weights)
    
    credits = [0] * K
    chosen_counts = [0] * K
    
    period_sequence = []
    cumulative_counts_at_step = []
    
    # Simulate one full period of length P = sum(weights)
    for _ in range(P):
        # 1. Add weights to credits
        for i in range(K):
            credits[i] += weights[i]
            
        # 2. Choose dataset with the largest credit (smallest index on tie)
        best_i = 0
        max_cred = credits[0]
        for i in range(1, K):
            if credits[i] > max_cred:
                max_cred = credits[i]
                best_i = i
                
        # 3. Subtract sum(weights) from the chosen dataset's credit
        credits[best_i] -= P
        
        period_sequence.append(best_i)
        cumulative_counts_at_step.append(chosen_counts[best_i])
        
        # 4. Increment choice count
        chosen_counts[best_i] += 1
        
    # Generate requested items from start_offset up to num_samples - 1
    result = []
    for t in range(start_offset, num_samples):
        q = t // P
        r = t % P
        ds_idx = period_sequence[r]
        total_chosen = q * weights[ds_idx] + cumulative_counts_at_step[r]
        ds = datasets[ds_idx]
        result.append(ds[total_chosen % len(ds)])
        
    return result