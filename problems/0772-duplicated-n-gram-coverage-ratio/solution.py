def dup_ngram_ratio(text: str, n: int) -> float:
    tokens = text.split()
    if len(tokens) < n:
        return 0.0

    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    total = len(ngrams)

    from collections import Counter
    counts = Counter(ngrams)

    duplicated = sum(1 for ng in ngrams if counts[ng] > 1)
    return round(duplicated / total, 4)