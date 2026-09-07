import heapq


def running_median(stream):
    """
    Computes the running median of a data stream using two heaps.

    Args:
        stream (list of numbers): Numbers arriving one at a time.

    Returns:
        list of float: Median after each number is added.
    """
    small = []  # max-heap stored as negative numbers
    large = []  # min-heap
    medians = []

    for num in stream:
        # Step 1: Add to max-heap (small)
        heapq.heappush(small, -float(num))

        # Step 2: Ensure max of small <= min of large
        if small and large and (-small[0] > large[0]):
            val = -heapq.heappop(small)
            heapq.heappush(large, val)

        # Step 3: Balance sizes (small can have at most 1 more element than large)
        if len(small) > len(large) + 1:
            val = -heapq.heappop(small)
            heapq.heappush(large, val)
        elif len(large) > len(small):
            val = heapq.heappop(large)
            heapq.heappush(small, -val)

        # Step 4: Compute median
        if len(small) > len(large):
            medians.append(float(-small[0]))
        else:
            medians.append(float(-small[0] + large[0]) / 2.0)

    return medians