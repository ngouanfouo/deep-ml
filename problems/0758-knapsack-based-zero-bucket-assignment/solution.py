def assign_buckets(sizes: list[int], n_ranks: int) -> list[int]:
    """
    Assign each matrix to a rank using LPT greedy heuristic.
    """
    # Sort by size descending, stable (preserves original order for ties)
    items = sorted(enumerate(sizes), key=lambda x: x[1], reverse=True)

    assignment = [0] * len(sizes)
    loads = [0] * n_ranks

    for idx, size in items:
        # Pick the rank with smallest load; tie → smallest index
        rank = min(range(n_ranks), key=lambda r: (loads[r], r))
        assignment[idx] = rank
        loads[rank] += size

    return assignment