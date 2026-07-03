import math

def softmax(scores: list[float]) -> list[float]:
    # Subtract max for numerical stability
    max_score = max(scores)
    exp_scores = [math.exp(score - max_score) for score in scores]
    sum_exp = sum(exp_scores)
    return [exp / sum_exp for exp in exp_scores]