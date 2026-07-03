import math

def binomial_probability(n: int, k: int, p: float) -> float:
    """
    Calculate the probability of exactly k successes in n Bernoulli trials.
    
    Args:
        n: Total number of trials
        k: Number of successes
        p: Probability of success on each trial
    
    Returns:
        Probability of k successes rounded to 5 decimal places
    """
    # Edge case: If asked for more successes than trials, the probability is 0
    if k < 0 or k > n:
        return 0.0
        
    # Calculate the binomial coefficient C(n, k)
    binomial_coefficient = math.comb(n, k)
    
    # Apply the binomial distribution formula
    probability = binomial_coefficient * (p ** k) * ((1 - p) ** (n - k))
    
    # Round to 5 decimal places as required by the example
    return round(probability, 5)

# --- Verification with the example ---
