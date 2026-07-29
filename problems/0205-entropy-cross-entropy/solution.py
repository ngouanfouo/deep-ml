import numpy as np

def entropy_and_cross_entropy(P: list[float], Q: list[float]) -> tuple[float, float]:
    """
    Compute entropy of P and cross-entropy between P and Q.
    
    Args:
        P: True probability distribution
        Q: Predicted probability distribution
    
    Returns:
        Tuple of (entropy H(P), cross-entropy H(P,Q))
    """
    # Convert to numpy arrays for easier computation
    P = np.array(P, dtype=np.float64)
    Q = np.array(Q, dtype=np.float64)
    
    # Normalize to ensure they sum to 1 (just in case)
    P = P / np.sum(P)
    Q = Q / np.sum(Q)
    
    # Entropy H(P) = -sum(P * log(P))
    # Only consider probabilities > 0 to avoid log(0)
    mask_P = P > 0
    entropy = -np.sum(P[mask_P] * np.log(P[mask_P]))
    
    # Cross-entropy H(P,Q) = -sum(P * log(Q))
    # Only consider probabilities where both P and Q are > 0
    mask_Q = Q > 0
    mask_both = mask_P & mask_Q
    cross_entropy = -np.sum(P[mask_both] * np.log(Q[mask_both]))
    
    return (float(entropy), float(cross_entropy))