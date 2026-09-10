def random_pick_with_weight(weights, r):
    # weights: list of positive integers
    # r: float in [0, 1)
    # return the selected index (int)
    total = sum(weights)
    target = r * total
    
    cumulative = 0
    for i, w in enumerate(weights):
        cumulative += w
        if cumulative > target:
            return i
    
    # Unreachable for valid inputs: r < 1 implies target < total,
    # so some cumulative sum must exceed target.
    return len(weights) - 1