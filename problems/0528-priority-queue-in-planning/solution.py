import numpy as np
import heapq

def priority_queue_planning(Q, model, predecessors, gamma, theta, max_steps):
    """
    Perform planning updates using a priority queue based on Bellman error.
    
    Args:
        Q: Initial Q-table, shape (num_states, num_actions)
        model: dict mapping (state, action) -> (reward, next_state)
        predecessors: dict mapping state -> list of (state, action) that transition to it
        gamma: float, discount factor
        theta: float, priority threshold (only update if error > theta)
        max_steps: int, maximum number of planning updates
    
    Returns:
        Updated Q-table as a list of lists, rounded to 4 decimal places
    """
    Q = np.array(Q, dtype=float, copy=True)

    def compute_error_and_target(s, a):
        """Compute the Bellman error and the Bellman target for (s, a)."""
        reward, next_state = model[(s, a)]
        if next_state == -1:
            target = float(reward)
        else:
            target = float(reward) + gamma * float(np.max(Q[next_state]))
        error = abs(target - Q[s, a])
        return error, target

    # ---- 1. Initialize the priority queue ----
    pq = []
    counter = 0  # tie-breaker so heapq never compares (s, a) tuples
    for (s, a) in model:
        error, _ = compute_error_and_target(s, a)
        if error > theta:
            heapq.heappush(pq, (-error, counter, s, a))
            counter += 1

    # ---- 2. Planning loop ----
    steps = 0
    while pq and steps < max_steps:
        neg_err, _, s, a = heapq.heappop(pq)

        # Stale-entry check: recompute error against the current Q-table
        error, target = compute_error_and_target(s, a)
        if error <= theta:
            continue

        # Bellman update (full backup for the deterministic model)
        Q[s, a] = target
        steps += 1

        # ---- 3. Propagate to predecessors of state s ----
        for (ps, pa) in predecessors.get(s, []):
            if (ps, pa) not in model:
                continue
            p_error, _ = compute_error_and_target(ps, pa)
            if p_error > theta:
                heapq.heappush(pq, (-p_error, counter, ps, pa))
                counter += 1

    return np.round(Q, 4).tolist()