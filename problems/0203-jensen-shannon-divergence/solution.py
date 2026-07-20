import numpy as np

def jensen_shannon_divergence(P: list[float], Q: list[float]) -> float:
    """
    Compute the Jensen-Shannon Divergence between two probability distributions.
    
    Args:
        P: First probability distribution
        Q: Second probability distribution
    
    Returns:
        Jensen-Shannon Divergence value as a float
    """
    p_arr = np.array(P, dtype=np.float64)
    q_arr = np.array(Q, dtype=np.float64)
    
    # 1. Normalize arrays to ensure they form valid probability distributions
    p_arr /= np.sum(p_arr)
    q_arr /= np.sum(q_arr)
    
    # 2. Compute the midpoint distribution M
    m_arr = 0.5 * (p_arr + q_arr)
    
    # Helper function to compute Kullback-Leibler divergence with stability guards
    def kl_divergence(dist_a, dist_m):
        # Mask out locations where dist_a is zero, since 0 * log(0) = 0
        mask = dist_a > 0
        return np.sum(dist_a[mask] * np.log(dist_a[mask] / dist_m[mask]))
        
    # 3. Compute JSD as the average of the two KL divergences
    jsd_val = 0.5 * kl_divergence(p_arr, m_arr) + 0.5 * kl_divergence(q_arr, m_arr)
    
    return float(jsd_val)