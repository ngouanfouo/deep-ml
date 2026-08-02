import numpy as np

def n_step_tree_backup(Q, states, actions, rewards, target_policy, terminal_states, n, alpha, gamma):
    """
    Implement the n-step tree backup algorithm for a single episode trajectory.
    
    Args:
        Q: np.ndarray of shape (num_states, num_actions), initial Q-values
        states: list of int, states visited (length T+1)
        actions: list of int, actions taken (length T)
        rewards: list of float, rewards received (length T)
        target_policy: np.ndarray of shape (num_states, num_actions), target policy pi(a|s)
        terminal_states: list of int, terminal state indices
        n: int, number of backup steps
        alpha: float, learning rate
        gamma: float, discount factor
    
    Returns:
        Q: np.ndarray of shape (num_states, num_actions), updated Q-values rounded to 4 decimals
    """
    Q = Q.copy().astype(float)
    T = len(actions)
    
    # Process each time step t along the trajectory sequence
    for t in range(T):
        tau = min(t + n, T)
        s_tau = states[tau]
        
        # 1. Compute the leaf value at tau
        if s_tau in terminal_states:
            G = 0.0
        else:
            # Expected value under the target policy using current Q-values
            G = np.sum(target_policy[s_tau] * Q[s_tau])
            
        # 2. Work backward from tau-1 down to t+1 to accumulate the branch return values
        for k in range(tau - 1, t, -1):
            s_k = states[k]
            a_k = actions[k]
            r_k = rewards[k]
            
            # Compute policy-weighted values for all non-taken actions at state s_k
            expected_non_taken = 0.0
            for a in range(Q.shape[1]):
                if a != a_k:
                    expected_non_taken += target_policy[s_k, a] * Q[s_k, a]
            
            # Combine the non-taken paths with the recursive continuation value along the taken path
            G = expected_non_taken + target_policy[s_k, a_k] * (r_k + gamma * G)
            
        # 3. Incorporate the final reward immediately succeeding state s_t
        target_return = rewards[t] + gamma * G
        
        # 4. Standard TD Update rule targeting the step return
        s_t = states[t]
        a_t = actions[t]
        Q[s_t, a_t] += alpha * (target_return - Q[s_t, a_t])
        
    return np.round(Q, 4)