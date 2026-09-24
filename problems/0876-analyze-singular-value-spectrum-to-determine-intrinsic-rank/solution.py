import numpy as np

def suggest_rank(delta_W: np.ndarray, energy_threshold: float) -> int:
    """
    Return the smallest rank k such that the top-k singular values of delta_W
    capture at least `energy_threshold` of the total squared-singular-value energy.
    """
    # Singular values only
    s = np.linalg.svd(delta_W, compute_uv=False)
    energy = s ** 2
    total_energy = energy.sum()

    # All-zero matrix has no energy
    if total_energy == 0:
        return 0

    cumulative_energy = np.cumsum(energy)
    target = energy_threshold * total_energy

    # First index where cumulative energy reaches the target
    k = int(np.searchsorted(cumulative_energy, target, side='left') + 1)
    return k