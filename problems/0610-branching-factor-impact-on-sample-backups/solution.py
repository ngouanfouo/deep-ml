import numpy as np

def branching_factor_backup_analysis(mdp: dict, values: list, gamma: float, n_samples: int, seed: int) -> dict:
    """
    Analyze how branching factor affects sample backup accuracy.
    """
    rng = np.random.RandomState(seed)
    values = np.asarray(values, dtype=float)

    # Group RMSEs by branching factor
    rmse_by_b = {}

    # Process state-action pairs in sorted order (by state, then action)
    for (state, action) in sorted(mdp.keys()):
        transitions = mdp[(state, action)]
        b = len(transitions)

        next_states = np.array([t[0] for t in transitions], dtype=int)
        probs = np.array([t[1] for t in transitions], dtype=float)
        rewards = np.array([t[2] for t in transitions], dtype=float)

        # Exact expected backup
        exact_q = float(np.sum(probs * (rewards + gamma * values[next_states])))

        # n_samples independent sample backups
        idx = rng.choice(b, size=n_samples, p=probs)
        sample_q = rewards[idx] + gamma * values[next_states[idx]]

        # RMSE of the sample backups relative to the exact Q-value
        rmse = float(np.sqrt(np.mean((sample_q - exact_q) ** 2)))

        rmse_by_b.setdefault(b, []).append(rmse)

    # Average RMSE per branching factor, rounded, keys ascending
    return {b: round(float(np.mean(rmse_by_b[b])), 4) for b in sorted(rmse_by_b)}