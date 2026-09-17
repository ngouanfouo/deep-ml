import torch
from collections import Counter
import math

def pass_at_1(responses_correct: torch.Tensor) -> float:
    """
    Compute pass@1 using PyTorch.
    
    Args:
        responses_correct: Boolean tensor
        
    Returns:
        pass@1 score
    """
    return responses_correct.float().mean().item()


def majority_voting(responses: list[str]) -> str:
    """
    Return most common response.
    """
    if not responses:
        return ""
    return Counter(responses).most_common(1)[0][0]


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Compute unbiased pass@k estimator.
    
    Args:
        n: total number of samples
        c: number of correct samples
        k: k in pass@k
        
    Returns:
        unbiased estimate of pass@k
    """
    if k > n:
        k = n
    if n - c < k:
        return 1.0
    # 1 - C(n-c, k) / C(n, k) computed via product to avoid overflow
    prod = 1.0
    for i in range(k):
        prod *= (n - c - i) / (n - i)
    return 1.0 - prod