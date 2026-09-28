import numpy as np

def expected_vs_sample_updates(P: np.ndarray, R: np.ndarray, gamma: float,
                               alpha: float, num_sweeps: int,
                               seed: int = 42) -> tuple:
    """
    Compare expected and sample updates for Q-value estimation.

    Expected updates are synchronous: each sweep computes every Q[s,a]
    using the Q-table snapshot from the start of the sweep.

    Sample updates are in-place: a single next state is drawn from the
    transition distribution and the Q-value is moved toward the sampled
    target with learning rate alpha, so later pairs in the same sweep
    see earlier updates.
    """
    np.random.seed(seed)

    num_states, num_actions, _ = P.shape

    # ---------- Expected updates (synchronous) ----------
    Q_expected = np.zeros((num_states, num_actions), dtype=float)

    for _ in range(num_sweeps):
        Q_old = Q_expected.copy()                     # snapshot for the sweep
        V_old = np.max(Q_old, axis=1)                 # (num_states,)
        Q_expected = R + gamma * np.einsum('san,n->sa', P, V_old)

    # ---------- Sample updates (in-place) ----------
    Q_sample = np.zeros((num_states, num_actions), dtype=float)

    for _ in range(num_sweeps):
        for s in range(num_states):
            for a in range(num_actions):
                # Draw a single next state from the transition distribution
                s_next = int(np.random.choice(num_states, p=P[s, a]))
                # Target from the *current* in-place Q-sample (may already be updated)
                best_next = float(np.max(Q_sample[s_next]))
                target = R[s, a] + gamma * best_next
                Q_sample[s, a] += alpha * (target - Q_sample[s, a])

    return (np.round(Q_expected, 4), np.round(Q_sample, 4))