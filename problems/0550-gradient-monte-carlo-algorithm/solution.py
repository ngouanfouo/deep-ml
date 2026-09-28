import numpy as np


def gradient_monte_carlo(episodes: list, alpha: float, gamma: float, n_epochs: int) -> list:
    """
    Learn value function weights using Gradient Monte Carlo with linear approximation.

    Args:
        episodes: List of episodes. Each episode is a list of (state_features, reward) tuples.
                  state_features is a list of floats, reward is a float.
        alpha: Learning rate (step size)
        gamma: Discount factor
        n_epochs: Number of passes through all episodes

    Returns:
        List of learned weight values, each rounded to 4 decimal places.
    """
    # Infer feature dimension from the first observed feature vector
    d = None
    for episode in episodes:
        if episode:
            d = len(episode[0][0])
            break
    if d is None:
        return []

    w = np.zeros(d, dtype=float)

    for _ in range(n_epochs):
        for episode in episodes:
            if not episode:
                continue
            T = len(episode)

            # Backward pass: discounted returns G_t
            returns = np.zeros(T, dtype=float)
            running = 0.0
            for t in range(T - 1, -1, -1):
                _, r = episode[t]
                running = r + gamma * running
                returns[t] = running

            # Forward pass: SGD updates using each state's return as target
            for t in range(T):
                x = np.asarray(episode[t][0], dtype=float)
                v_hat = float(np.dot(w, x))
                w = w + alpha * (returns[t] - v_hat) * x

    return [round(float(wi), 4) for wi in w]