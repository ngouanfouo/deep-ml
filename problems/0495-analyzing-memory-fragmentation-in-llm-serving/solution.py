def analyze_memory_fragmentation(block_status: list) -> dict:
    """
    Analyze memory fragmentation in a block-based memory pool.

    Args:
        block_status: List of ints where 1 = allocated block, 0 = free block

    Returns:
        dict with keys:
            'utilization': float, fraction of allocated blocks
            'num_free_fragments': int, count of contiguous free regions
            'largest_free_fragment': int, size of largest contiguous free region
            'fragmentation_ratio': float, measure of free memory scatter
    """
    n = len(block_status)
    allocated = sum(block_status)
    utilization = allocated / n if n > 0 else 0.0

    # Walk the pool and collect the sizes of contiguous free runs.
    free_fragments = []
    current_run = 0
    for b in block_status:
        if b == 0:
            current_run += 1
        else:
            if current_run > 0:
                free_fragments.append(current_run)
            current_run = 0
    if current_run > 0:
        free_fragments.append(current_run)

    num_free_fragments = len(free_fragments)
    largest_free_fragment = max(free_fragments) if free_fragments else 0
    total_free = sum(free_fragments)

    # Fragmentation: 0 when all free blocks are contiguous,
    # approaches 1 when free blocks are maximally scattered.
    if total_free == 0:
        fragmentation_ratio = 0.0
    else:
        fragmentation_ratio = 1.0 - largest_free_fragment / total_free

    return {
        'utilization': round(utilization, 4),
        'num_free_fragments': num_free_fragments,
        'largest_free_fragment': largest_free_fragment,
        'fragmentation_ratio': round(fragmentation_ratio, 4),
    }