import numpy as np

def q_lambda_watkins(episodes: list, n_states: int, n_actions: int, gamma: float, alpha: float, lam: float) -> list:
    """
    Implement Watkins's Q(lambda) with eligibility traces.
    
    Args:
        episodes: List of episodes, each a list of (state, action, reward, next_state, done) tuples
        n_states: Number of states
        n_actions: Number of actions
        gamma: Discount factor
        alpha: Learning rate
        lam: Lambda parameter for trace decay
    
    Returns:
        Q-value table as nested list of shape (n_states, n_actions), rounded to 4 decimal places
    """
    # Initialize action-value table to zeros
    Q = np.zeros((n_states, n_actions), dtype=np.float64)
    
    for episode in episodes:
        # Reset eligibility traces to zero at the start of each episode
        E = np.zeros((n_states, n_actions), dtype=np.float64)
        
        # Process the transition sequence within the episode sequentially
        for i, (state, action, reward, next_state, done) in enumerate(episode):
            # 1. Determine the greedy action at the next state (breaking ties with the lowest index)
            next_qs = Q[next_state]
            max_q_next = np.max(next_qs)
            greedy_action_next = int(np.min(np.where(next_qs == max_q_next)[0]))
            
            # 2. Compute target value and temporal difference (TD) error
            if done:
                q_target = reward
            else:
                q_target = reward + gamma * max_q_next
                
            delta = q_target - Q[state, action]
            
            # 3. Accumulating trace increment
            E[state, action] += 1.0
            
            # 4. Global Q-value table update via the active traces
            Q += alpha * delta * E
            
            # 5. Check trace cutting for the next transition lookahead
            # If there is a next transition within this pre-collected episode sequence,
            # look ahead at the actual action taken by the behavior policy.
            if i + 1 < len(episode):
                next_action_taken = episode[i + 1][1]
                if next_action_taken == greedy_action_next:
                    # Next action is greedy: decay the traces normally
                    E *= gamma * lam
                else:
                    # Next action is exploratory (non-greedy): cut the traces
                    E.fill(0.0)
            else:
                # End of the episode: decay traces normally (or let the loop terminate)
                E *= gamma * lam

    # Return the Q-table converted to a nested list rounded to 4 decimal places
    return [[round(float(val), 4) for val in row] for row in Q]