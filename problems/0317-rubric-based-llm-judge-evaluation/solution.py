import torch


def rubric_llm_judge_evaluation(
    judge_scores: list[list[float]],
    criteria_weights: list[float],
    passing_threshold: float = 0.6,
    max_score: float = 5.0,
) -> dict:
    """Evaluate LLM response using rubric-based multi-judge scoring.

    Args:
        judge_scores: 2D list where judge_scores[i][j] is judge i's score for
          criterion j
        criteria_weights: Weights for each criterion (should sum to 1)
        passing_threshold: Minimum normalized score to pass (0 to 1)
        max_score: Maximum possible score for each criterion

    Returns:
        Dictionary with evaluation results containing:
        - weighted_score: Overall weighted score
        - normalized_score: Score normalized to 0-1 scale
        - criterion_scores: Average score per criterion across judges
        - pass_status: Boolean indicating if response passes threshold
        - judge_agreement: Agreement metric from 0 to 1
    """
    scores_tensor = torch.tensor(judge_scores, dtype=torch.float64)
    weights_tensor = torch.tensor(criteria_weights, dtype=torch.float64)

    # 1. Criterion scores: Average score per criterion across judges
    criterion_scores_tensor = torch.mean(scores_tensor, dim=0)

    # 2. Weighted score and normalized score
    weighted_score_tensor = torch.sum(criterion_scores_tensor * weights_tensor)
    normalized_score_tensor = weighted_score_tensor / max_score

    # 3. Pass status
    pass_status = bool(normalized_score_tensor.item() >= passing_threshold)

    # 4. Judge agreement: uses population standard deviation (correction=0)
    n_judges = scores_tensor.shape[0]
    if n_judges > 1:
        std_per_criterion = torch.std(scores_tensor, dim=0, correction=0)
        avg_std = torch.mean(std_per_criterion)
        max_std = max_score / 2.0
        judge_agreement_tensor = torch.clamp(
            1.0 - (avg_std / max_std), 0.0, 1.0
        )
    else:
        judge_agreement_tensor = torch.tensor(1.0, dtype=torch.float64)

    # Round to 4 decimal places
    criterion_scores = [
        round(val.item(), 4) for val in criterion_scores_tensor
    ]
    weighted_score = round(weighted_score_tensor.item(), 4)
    normalized_score = round(normalized_score_tensor.item(), 4)
    judge_agreement = round(judge_agreement_tensor.item(), 4)

    return {
        "weighted_score": weighted_score,
        "normalized_score": normalized_score,
        "criterion_scores": criterion_scores,
        "pass_status": pass_status,
        "judge_agreement": judge_agreement,
    }