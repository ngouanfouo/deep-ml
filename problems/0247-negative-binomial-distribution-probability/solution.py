import math

def negative_binomial_pmf(k: int, r: int, p: float) -> float:
    """
    Calculate the probability of observing exactly k failures
    before achieving r successes in independent Bernoulli trials.
    
    Args:
        k: Number of failures (non-negative integer)
        r: Number of successes required (positive integer)
        p: Probability of success on each trial (0 < p <= 1)
    
    Returns:
        Probability P(X = k) rounded to 5 decimal places
    """
    # Input validation
    if k < 0 or r <= 0 or p <= 0 or p > 1:
        return 0.0
    
    if p == 1.0:
        return 1.0 if k == 0 else 0.0
    
    # Using math.comb for integer combinations
    # Number of ways to arrange k failures and r-1 successes
    # (the r-th success is fixed as the last trial)
    combinations = math.comb(k + r - 1, k)
    
    # PMF formula
    prob = combinations * (p ** r) * ((1 - p) ** k)
    
    return round(prob, 5)