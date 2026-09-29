import math
from collections import Counter

def ngram_keep_probabilities(samples, n, threshold):
    """Compute per-sample keep probabilities from n-gram frequencies."""
    sample_ngrams = []
    global_counts = Counter()

    for sample in samples:
        tokens = sample.lower().split()
        if len(tokens) < n:
            grams = []
        else:
            grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
        sample_ngrams.append(grams)
        global_counts.update(grams)

    probs = []
    for grams in sample_ngrams:
        if not grams:
            probs.append(1.0)
            continue

        best = 0.0
        for g in grams:
            f = global_counts[g]
            score = min(1.0, math.sqrt(threshold / f))
            if score > best:
                best = score

        probs.append(round(best, 4))

    return probs