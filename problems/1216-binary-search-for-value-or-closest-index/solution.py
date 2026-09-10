def binary_search_closest(arr, target):
    if not arr:
        return -1
    
    lo, hi = 0, len(arr) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    
    # lo is the leftmost index with arr[lo] >= target
    # Check candidates: lo and lo-1
    candidates = []
    if lo < len(arr):
        candidates.append(lo)
    if lo - 1 >= 0:
        candidates.append(lo - 1)
    
    best = candidates[0]
    for c in candidates[1:]:
        d_c = abs(arr[c] - target)
        d_best = abs(arr[best] - target)
        if d_c < d_best:
            best = c
        elif d_c == d_best and arr[c] < arr[best]:
            best = c
    return best