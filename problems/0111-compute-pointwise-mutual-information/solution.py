import numpy as np

def compute_pmi(joint_counts: int, total_counts_x: int, total_counts_y: int, total_samples: int) -> float:
    """
    Computes the Pointwise Mutual Information (PMI) of two events.
    
    Args:
        joint_counts: Number of times x and y co-occur
        total_counts_x: Total occurrences of event x
        total_counts_y: Total occurrences of event y
        total_samples: Total number of observations/samples in the dataset
        
    Returns:
        float: PMI score rounded to 3 decimal places. Returns -inf or nan if joint count is 0.
    """
    # Handle the case where events never co-occur to prevent division by zero or log of 0
    if joint_counts == 0:
        return float('-inf')
        
    # Calculate probabilities
    p_xy = joint_counts / total_samples
    p_x = total_counts_x / total_samples
    p_y = total_counts_y / total_samples
    
    # Compute PMI using base-2 logarithm
    pmi = np.log2(p_xy / (p_x * p_y))
    
    return round(float(pmi), 3)