import numpy as np

def gaussian_policy(state: np.ndarray, mean_weights: np.ndarray, std: float, action: float) -> dict:
    """
    Compute Gaussian policy outputs for continuous control.

    Args:
        state: numpy array of shape (d,) representing state features
        mean_weights: numpy array of shape (d,) - policy parameters for the mean
        std: float, fixed standard deviation of the policy
        action: float, the action value to evaluate

    Returns:
        Dictionary with keys 'mean', 'pdf', 'log_prob', 'score'
    """
    state = np.asarray(state, dtype=float)
    mean_weights = np.asarray(mean_weights, dtype=float)

    # Mean: linear function of state features
    mean = float(state @ mean_weights)

    # PDF of the action under N(mean, std^2)
    deviation = action - mean
    pdf = float(
        (1.0 / (std * np.sqrt(2.0 * np.pi)))
        * np.exp(-0.5 * (deviation / std) ** 2)
    )

    # Log-probability
    log_prob = float(
        -0.5 * (deviation / std) ** 2
        - np.log(std)
        - 0.5 * np.log(2.0 * np.pi)
    )

    # Score: gradient of log_prob w.r.t. mean_weights
    # d/dw log N(a | w^T s, sigma^2) = ((a - w^T s) / sigma^2) * s
    score = ((deviation / (std ** 2)) * state).tolist()

    return {
        'mean': round(mean, 4),
        'pdf': round(pdf, 4),
        'log_prob': round(log_prob, 4),
        'score': [round(float(x), 4) for x in score],
    }