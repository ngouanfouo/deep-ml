import numpy as np

def hamming_distance_kanerva(state: list, prototypes: list, threshold: int) -> tuple:
    """
    Compute Hamming distances and find active prototypes for Kanerva coding.

    Args:
        state: Binary state vector (list of 0s and 1s).
        prototypes: List of binary prototype vectors.
        threshold: Maximum Hamming distance for a prototype to be active.

    Returns:
        Tuple of (distances, active_indices).
    """
    state = np.asarray(state, dtype=int)
    prototypes = np.asarray(prototypes, dtype=int)
    
    # Hamming distance = number of positions where bits differ
    # XOR gives 1 where bits differ; sum along axis 1 counts them
    distances = np.sum(state[None, :] != prototypes, axis=1)
    
    # Active prototypes: distance <= threshold
    active_indices = np.where(distances <= threshold)[0].tolist()
    
    return (distances.tolist(), active_indices)