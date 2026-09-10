def _clean(intervals):
    """Sort and merge overlapping or touching closed intervals."""
    if not intervals:
        return []
    sorted_iv = sorted(intervals, key=lambda x: (x[0], x[1]))
    merged = [list(sorted_iv[0])]
    for s, e in sorted_iv[1:]:
        # Closed intervals: sharing an endpoint counts as touching → merge
        if s <= merged[-1][1]:
            if e > merged[-1][1]:
                merged[-1][1] = e
        else:
            merged.append([s, e])
    return merged


def interval_intersections(A, B):
    a = _clean(A)
    b = _clean(B)
    
    result = []
    i = j = 0
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0])
        hi = min(a[i][1], b[j][1])
        if lo <= hi:
            result.append([lo, hi])
        # Advance whichever interval ends first
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    
    return result


# Alias matching the alternative name in the stub
interval_intervals = interval_intersections