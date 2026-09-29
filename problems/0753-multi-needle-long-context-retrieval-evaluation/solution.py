import bisect

def evaluate_mrcr(tasks, buckets):
    """
    Compute per-bucket mean match rate for multi-needle retrieval tasks.

    Args:
        tasks: list of dicts with keys 'context_length', 'expected', 'predicted'.
        buckets: list of ints, sorted ascending.

    Returns:
        list of floats (one per bucket), each rounded to 4 decimals.
    """
    bucket_sums = [0.0] * len(buckets)
    bucket_counts = [0] * len(buckets)

    for task in tasks:
        c = task['context_length']
        expected = task['expected']
        predicted = task['predicted']

        # Find the smallest bucket >= c
        idx = bisect.bisect_left(buckets, c)
        if idx >= len(buckets):
            continue  # larger than every bucket -> drop

        # Compute match rate
        if len(expected) == 0:
            match_rate = 0.0
        else:
            expected_set = set(expected)
            predicted_set = set(predicted)
            matched = len(expected_set & predicted_set)
            match_rate = matched / len(expected_set)

        bucket_sums[idx] += match_rate
        bucket_counts[idx] += 1

    result = []
    for i in range(len(buckets)):
        if bucket_counts[i] == 0:
            result.append(0.0)
        else:
            result.append(round(bucket_sums[i] / bucket_counts[i], 4))

    return result