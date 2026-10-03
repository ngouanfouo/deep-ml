import numpy as np

def directional_stripe_tiling(states: list, bounds: tuple, num_stripes: int, num_tilings: int) -> tuple:
    states = np.asarray(states, dtype=float)
    n_states = len(states)
    total_features = num_tilings * 2 * num_stripes

    lows = np.array([bounds[0][0], bounds[1][0]], dtype=float)
    highs = np.array([bounds[0][1], bounds[1][1]], dtype=float)
    widths = (highs - lows) / num_stripes          # stripe width per dimension

    feature_matrix = np.zeros((n_states, total_features), dtype=int)
    active_indices = []

    for i, s in enumerate(states):
        active = []
        for t in range(num_tilings):
            for d in range(2):
                offset = t * widths[d] / num_tilings
                idx = int(np.floor((s[d] - lows[d] + offset) / widths[d]))
                idx = max(0, min(idx, num_stripes - 1))
                base = t * (2 * num_stripes) + d * num_stripes
                active.append(base + idx)
        active = sorted(active)
        active_indices.append(active)
        feature_matrix[i, active] = 1

    return feature_matrix, active_indices