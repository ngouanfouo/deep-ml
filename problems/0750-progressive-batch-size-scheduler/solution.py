from bisect import bisect_right

def progressive_batch_size(queries, milestones, batch_sizes):
    """
    Return the active batch size for each query token count.

    Args:
        queries: list of token counts (ints)
        milestones: list of increasing token counts
        batch_sizes: list of batch sizes; length = len(milestones) + 1

    Returns:
        list of batch sizes (ints), one per query
    """
    return [batch_sizes[bisect_right(milestones, q)] for q in queries]