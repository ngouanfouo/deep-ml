import numpy as np

def generalized_policy_iteration(
    num_states: int,
    num_actions: int,
    transitions: np.ndarray,
    rewards: np.ndarray,
    discount: float,
    num_iterations: int,
    eval_sweeps: int,
    terminal_states: list
) -> dict:
    """
    Simulate Generalized Policy Iteration on a finite MDP.
    
    Args:
        num_states: Number of states in the MDP
        num_actions: Number of actions available
        transitions: Shape (S, A, S) transition probabilities
        rewards: Shape (S, A, S) reward function
        discount: Discount factor gamma in [0, 1]
        num_iterations: Number of GPI cycles (eval + improve)
        eval_sweeps: Number of synchronous policy evaluation sweeps per cycle
        terminal_states: List of terminal state indices
    
    Returns:
        Dictionary with:
            'values': List of floats (rounded to 4 decimal places)
            'policy': List of ints (action for each state)
    """
    terminal_set = set(terminal_states)

    # Initialize value function and policy
    V = np.zeros(num_states, dtype=float)
    policy = np.zeros(num_states, dtype=int)  # all actions initially 0

    for _ in range(num_iterations):
        # ---------- Policy Evaluation ----------
        for _ in range(eval_sweeps):
            new_V = np.zeros(num_states, dtype=float)
            for s in range(num_states):
                if s in terminal_set:
                    new_V[s] = 0.0
                    continue
                a = policy[s]
                # Bellman expectation backup for the current policy action
                value = 0.0
                for s_next in range(num_states):
                    p = transitions[s, a, s_next]
                    if p == 0.0:
                        continue
                    r = rewards[s, a, s_next]
                    value += p * (r + discount * V[s_next])
                new_V[s] = value
            V = new_V

        # ---------- Policy Improvement ----------
        for s in range(num_states):
            if s in terminal_set:
                continue
            q_values = np.zeros(num_actions, dtype=float)
            for a in range(num_actions):
                q = 0.0
                for s_next in range(num_states):
                    p = transitions[s, a, s_next]
                    if p == 0.0:
                        continue
                    r = rewards[s, a, s_next]
                    q += p * (r + discount * V[s_next])
                q_values[a] = q
            # np.argmax returns the first occurrence of the maximum -> smallest index on ties
            policy[s] = int(np.argmax(q_values))

    # Prepare output
    values_out = np.round(V, 4).tolist()
    policy_out = policy.tolist()

    return {
        'values': values_out,
        'policy': policy_out
    }