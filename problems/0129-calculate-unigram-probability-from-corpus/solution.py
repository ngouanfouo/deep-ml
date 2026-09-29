def unigram_probability(corpus: str, word: str) -> float:
    tokens = corpus.split()
    total = len(tokens)
    if total == 0:
        return 0.0
    count = tokens.count(word)
    return round(count / total, 4)