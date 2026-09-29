def detect_contamination(test_examples, train_corpus, n, threshold):
    """
    Args:
        test_examples: list[str] - test set documents
        train_corpus:  list[str] - training set documents
        n: int - n-gram size
        threshold: float - coverage ratio in [0, 1] above which an example is flagged

    Returns:
        float - contamination percentage in [0, 100]
    """
    if n <= 0:
        return 0.0

    # Collect all n-grams from the training corpus
    train_ngrams = set()
    for doc in train_corpus:
        tokens = doc.split()
        for i in range(len(tokens) - n + 1):
            train_ngrams.add(tuple(tokens[i:i + n]))

    if not test_examples:
        return 0.0

    contaminated = 0

    for example in test_examples:
        tokens = example.split()
        L = len(tokens)

        if L == 0:
            continue  # not contaminated

        if L < n:
            coverage = 0.0
        else:
            covered = [False] * L
            for i in range(L - n + 1):
                window = tuple(tokens[i:i + n])
                if window in train_ngrams:
                    for j in range(i, i + n):
                        covered[j] = True

            coverage = sum(covered) / L

        if coverage >= threshold:
            contaminated += 1

    return (contaminated / len(test_examples)) * 100.0