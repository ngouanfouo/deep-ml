import numpy as np

def mountain_car_sarsa(
    n_episodes: int,
    n_bins: int = 8,
    alpha: float = 0.05,
    gamma: float = 1.0,
    epsilon: float = 0.1,
    max_steps: int = 200,
    seed: int = 42,
) -> tuple:
    """
    Run episodic semi-gradient Sarsa on the Mountain Car problem.

    Args:
        n_episodes: Number of episodes to run
        n_bins: Number of bins per state dimension for feature construction
        alpha: Step size (learning rate)
        gamma: Discount factor
        epsilon: Exploration rate for epsilon-greedy policy
        max_steps: Maximum steps per episode
        seed: Random seed for reproducibility

    Returns:
        Tuple of (episode_lengths, weight_sum):
            - episode_lengths: list of int, steps taken per episode
            - weight_sum: float, sum of final weight vector rounded to 4 decimal places
    """
    rng = np.random.RandomState(seed)

    # Environment constants
    POS_MIN, POS_MAX = -1.2, 0.5
    VEL_MIN, VEL_MAX = -0.07, 0.07
    N_ACTIONS = 3

    # Feature dimensions
    n_features = n_bins * n_bins * N_ACTIONS
    w = np.zeros(n_features)

    # Precompute bin edges (excluding the leftmost edge)
    pos_edges = np.linspace(POS_MIN, POS_MAX, n_bins + 1)[1:]
    vel_edges = np.linspace(VEL_MIN, VEL_MAX, n_bins + 1)[1:]

    def get_feature(position: float, velocity: float, action: int) -> int:
        """Return the index of the single active one-hot feature."""
        pos_bin = np.searchsorted(pos_edges, position)
        vel_bin = np.searchsorted(vel_edges, velocity)
        pos_bin = np.clip(pos_bin, 0, n_bins - 1)
        vel_bin = np.clip(vel_bin, 0, n_bins - 1)
        return (pos_bin * n_bins + vel_bin) * N_ACTIONS + action

    def q_value(position: float, velocity: float, action: int) -> float:
        return w[get_feature(position, velocity, action)]

    def choose_action(position: float, velocity: float) -> int:
        """Epsilon-greedy action selection with ties broken by lowest action index."""
        if rng.rand() < epsilon:
            return rng.randint(N_ACTIONS)
        qs = [q_value(position, velocity, a) for a in range(N_ACTIONS)]
        return int(np.argmax(qs))  # np.argmax returns first occurrence (lowest index)

    episode_lengths = []

    for _ in range(n_episodes):
        # Sample start state
        position = rng.uniform(-0.6, -0.4)
        velocity = 0.0
        action = choose_action(position, velocity)

        steps = 0
        while True:
            steps += 1

            # Environment dynamics
            new_velocity = velocity + 0.001 * (action - 1) - 0.0025 * np.cos(3 * position)
            new_velocity = np.clip(new_velocity, VEL_MIN, VEL_MAX)
            new_position = position + new_velocity
            new_position = np.clip(new_position, POS_MIN, POS_MAX)

            if new_position <= POS_MIN:  # hit left boundary
                new_velocity = 0.0

            reward = -1.0
            terminal = new_position >= POS_MAX

            # Semi-gradient Sarsa update
            if terminal:
                target = reward
            else:
                next_action = choose_action(new_position, new_velocity)
                target = reward + gamma * q_value(new_position, new_velocity, next_action)

            feat = get_feature(position, velocity, action)
            w[feat] += alpha * (target - w[feat])

            if terminal:
                break

            position, velocity = new_position, new_velocity
            action = next_action

            if steps >= max_steps:
                break

        episode_lengths.append(steps)

    weight_sum = round(float(np.sum(w)), 4)
    return episode_lengths, weight_sum