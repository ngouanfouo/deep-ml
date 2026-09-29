def detect_contamination(training_docs: list, benchmark_examples: list, n: int = 8, threshold: float = 0.8) -> dict:
    def tokenize(text: str) -> list:
        return text.lower().split()

    def get_ngrams(tokens: list, n: int) -> set:
        if len(tokens) < n:
            return set()
        return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}

    # Build the set of all n-grams present anywhere in the training corpus
    training_ngrams = set()
    for doc in training_docs:
        training_ngrams |= get_ngrams(tokenize(doc), n)

    scores = []
    flagged = []

    for example in benchmark_examples:
        example_ngrams = get_ngrams(tokenize(example), n)
        if not example_ngrams:
            score = 0.0
        else:
            overlap = sum(1 for ng in example_ngrams if ng in training_ngrams)
            score = overlap / len(example_ngrams)
        scores.append(round(score, 4))
        flagged.append(score >= threshold)

    total = len(benchmark_examples)
    contamination_rate = round(sum(flagged) / total, 4) if total > 0 else 0.0

    return {
        'scores': scores,
        'flagged': flagged,
        'contamination_rate': contamination_rate,
    }