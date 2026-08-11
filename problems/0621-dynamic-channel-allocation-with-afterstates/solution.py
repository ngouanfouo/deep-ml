import numpy as np


def channel_afterstate_step(
    allocation: np.ndarray,
    call_cell: int,
    adjacency: np.ndarray,
    V: dict,
    alpha: float,
    gamma: float,
    prev_afterstate_key,
    prev_reward: float,
) -> dict:
    """Perform one step of dynamic channel allocation using afterstate values.

    Args:
        allocation: binary array (n_cells, n_channels), 1 = channel in use
        call_cell: cell index requesting a channel
        adjacency: binary array (n_cells, n_cells), 1 = cells are adjacent
        V: dict mapping afterstate keys (tuple of tuples) to float values
        alpha: learning rate
        gamma: discount factor
        prev_afterstate_key: key of previous afterstate (None if first step)
        prev_reward: reward received at previous step

    Returns:
        dict with 'chosen_channel', 'reward', 'afterstate_key', 'V'
    """
    n_cells, n_channels = allocation.shape

    def get_key(mat):
        return tuple(tuple(int(x) for x in row) for row in mat)

    candidate_actions = []

    # 1. Candidate action: Rejection
    rejection_afterstate = allocation.copy()
    rejection_key = get_key(rejection_afterstate)
    rejection_reward = 0.0
    rejection_val = rejection_reward + gamma * float(V.get(rejection_key, 0.0))
    candidate_actions.append(
        {
            "channel": -1,
            "reward": rejection_reward,
            "afterstate_key": rejection_key,
            "value": rejection_val,
            "is_accept": False,
        }
    )

    # 2. Candidate actions: Accept on free channels
    for ch in range(n_channels):
        if allocation[call_cell, ch] == 0:
            additional_interference = 0
            for v in range(n_cells):
                if (
                    v != call_cell
                    and adjacency[call_cell, v] == 1
                    and allocation[v, ch] == 1
                ):
                    additional_interference += 1

            accept_reward = 1.0 - float(additional_interference)

            accept_afterstate = allocation.copy()
            accept_afterstate[call_cell, ch] = 1
            accept_key = get_key(accept_afterstate)

            accept_val = accept_reward + gamma * float(V.get(accept_key, 0.0))
            candidate_actions.append(
                {
                    "channel": ch,
                    "reward": accept_reward,
                    "afterstate_key": accept_key,
                    "value": accept_val,
                    "is_accept": True,
                }
            )

    # Find maximum expected value with numerical tolerance for tie-breaking
    max_val = max(act["value"] for act in candidate_actions)
    best_actions = [
        act for act in candidate_actions if abs(act["value"] - max_val) < 1e-9
    ]

    # Tie-breaking logic: prefer acceptance over rejection, then lowest channel index
    accepting_best = [act for act in best_actions if act["is_accept"]]
    if accepting_best:
        chosen_action = min(accepting_best, key=lambda act: act["channel"])
    else:
        chosen_action = best_actions[0]

    chosen_channel = chosen_action["channel"]
    chosen_reward = chosen_action["reward"]
    current_afterstate_key = chosen_action["afterstate_key"]

    # TD(0) update on previous afterstate if present
    if prev_afterstate_key is not None:
        v_prev = float(V.get(prev_afterstate_key, 0.0))
        v_current = float(V.get(current_afterstate_key, 0.0))
        td_target = prev_reward + gamma * v_current
        new_v_prev = v_prev + alpha * (td_target - v_prev)
        V[prev_afterstate_key] = new_v_prev

    # Ensure all values in V are rounded to 4 decimal places
    for k in list(V.keys()):
        V[k] = round(float(V[k]), 4)

    return {
        "chosen_channel": int(chosen_channel),
        "reward": round(float(chosen_reward), 4),
        "afterstate_key": current_afterstate_key,
        "V": V,
    }