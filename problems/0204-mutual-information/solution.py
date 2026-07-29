import numpy as np

def mutual_information(joint_prob: list[list[float]]) -> float:
    """
    Compute the mutual information between two random variables in nats (base e).
    
    Args:
        joint_prob: 2D joint probability distribution P(X,Y) (should sum to 1)
    
    Returns:
        Mutual information I(X;Y) in nats
    """
    P_xy = np.array(joint_prob, dtype=np.float64)
    # Do NOT normalize - assume input is already a valid probability distribution
    
    P_x = np.sum(P_xy, axis=1)
    P_y = np.sum(P_xy, axis=0)
    
    P_x_times_P_y = np.outer(P_x, P_y)
    
    mask = P_xy > 0
    
    # Use natural log (base e) to get values in nats
    mi_val = np.sum(P_xy[mask] * np.log(P_xy[mask] / P_x_times_P_y[mask]))
    
    return float(mi_val)