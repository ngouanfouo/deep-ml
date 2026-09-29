import re

def evaluate_corpus_qa(corpus_length, questions):
    """
    Evaluate model predictions on a long-context QA benchmark.

    Args:
        corpus_length: int, total number of tokens in the corpus.
        questions: list of dicts with keys 'gold', 'pred', 'position'.

    Returns:
        [overall_em, early_acc, mid_acc, late_acc] as a list of floats.
    """
    def normalize(text: str) -> str:
        text = text.lower()
        # Remove articles as whole words
        text = re.sub(r'\b(a|an|the)\b', ' ', text)
        # Replace every char that is not lowercase letter, digit, or whitespace with space
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        # Collapse whitespace and strip
        text = ' '.join(text.split())
        return text

    total = len(questions)
    if total == 0:
        return [0.0, 0.0, 0.0, 0.0]

    correct_total = 0
    early_correct = early_count = 0
    mid_correct = mid_count = 0
    late_correct = late_count = 0

    one_third = corpus_length / 3.0
    two_thirds = 2.0 * corpus_length / 3.0

    for q in questions:
        gold_norm = normalize(q['gold'])
        pred_norm = normalize(q['pred'])
        is_correct = (gold_norm == pred_norm)

        if is_correct:
            correct_total += 1

        pos = q['position']
        if pos < one_third:
            early_count += 1
            if is_correct:
                early_correct += 1
        elif pos < two_thirds:
            mid_count += 1
            if is_correct:
                mid_correct += 1
        else:
            late_count += 1
            if is_correct:
                late_correct += 1

    overall_em = correct_total / total
    early_acc = early_correct / early_count if early_count > 0 else 0.0
    mid_acc = mid_correct / mid_count if mid_count > 0 else 0.0
    late_acc = late_correct / late_count if late_count > 0 else 0.0

    return [overall_em, early_acc, mid_acc, late_acc]