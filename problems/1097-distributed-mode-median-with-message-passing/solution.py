import numpy as np

def distributed_stats(nodes, lo, hi):
    """
    Compute global mode and median across nodes using compact per-node summaries (histograms).

    Args:
        nodes: list of lists of integers, each integer in [lo, hi]
        lo: int, lower bound of measurement values
        hi: int, upper bound of measurement values

    Returns:
        tuple (mode, median): mode is an int, median is a float
    """
    range_size = hi - lo + 1
    global_hist = np.zeros(range_size, dtype=int)
    
    # 1. Aggregate per-node summaries (histograms)
    for node in nodes:
        if node:
            # Count occurrences within the valid range for each node
            hist, _ = np.histogram(node, bins=range_size, range=(lo - 0.5, hi + 0.5))
            global_hist += hist
            
    total_count = np.sum(global_hist)
    
    # 2. Compute global mode
    # Find the maximum frequency, and pick the smallest value (first occurrence) in case of ties
    max_freq = np.max(global_hist)
    mode_index = np.argmax(global_hist == max_freq)
    mode = int(lo + mode_index)
    
    # 3. Compute global median using cumulative distribution of the histogram
    cum_hist = np.cumsum(global_hist)
    
    if total_count % 2 == 1:
        # Odd number of elements: single middle element at index total_count // 2
        mid_idx = total_count // 2
        median_index = np.searchsorted(cum_hist, mid_idx, side='right')
        median = float(lo + median_index)
    else:
        # Even number of elements: average of elements at indices (total_count // 2) - 1 and total_count // 2
        idx1 = (total_count // 2) - 1
        idx2 = total_count // 2
        
        med_idx1 = np.searchsorted(cum_hist, idx1, side='right')
        med_idx2 = np.searchsorted(cum_hist, idx2, side='right')
        
        val1 = lo + med_idx1
        val2 = lo + med_idx2
        median = float(val1 + val2) / 2.0
        
    return (mode, median)