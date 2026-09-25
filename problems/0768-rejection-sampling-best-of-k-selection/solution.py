def rejection_sampling_best_of_k(candidates, scores):
    """
    Select the highest-scoring candidate per prompt.

    Args:
        candidates: list of N lists, each containing K candidate outputs.
        scores: list of N lists, each containing K reward scores.

    Returns:
        List of N selected candidates.
    """
    selected = []
    for prompt_candidates, prompt_scores in zip(candidates, scores):
        # argmax returns the first occurrence of the maximum value,
        # which naturally handles ties in favor of the lowest index.
        best_idx = max(range(len(prompt_scores)), key=lambda i: prompt_scores[i])
        selected.append(prompt_candidates[best_idx])
    return selected