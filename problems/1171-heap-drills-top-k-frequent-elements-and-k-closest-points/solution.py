from collections import Counter


def top_k_frequent(nums, k):
    """
    Returns the k most frequent integers from the input list, ordered by
    frequency descending, and by value ascending for ties.
    """
    counts = Counter(nums)
    # Sort by frequency descending (-count), then by value ascending (num)
    sorted_items = sorted(counts.keys(), key=lambda num: (-counts[num], num))
    return sorted_items[:k]


def k_closest_points(points, k):
    """
    Returns the k points nearest to the origin (0, 0) based on Euclidean distance,
    ordered by distance ascending, breaking ties by smaller x then smaller y.
    """
    # Sort using squared distance to avoid floating-point inaccuracies
    sorted_points = sorted(points, key=lambda p: (p[0]**2 + p[1]**2, p[0], p[1]))
    return sorted_points[:k]