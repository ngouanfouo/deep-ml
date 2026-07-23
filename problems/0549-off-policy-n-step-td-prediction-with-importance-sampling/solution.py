import numpy as np

def off_policy_nstep_td(
    episodes: list,
    behavior_policy: list,
    target_policy: list,
    num_states: int,
    num_actions: int,
    n: int,
    alpha: float,
    gamma: float
) -> np.ndarray:
    """
    Off-policy n-step TD prediction for state values using importance sampling.

    Args:
        episodes: List of episodes, each a list of (state, action, reward) tuples.
        behavior_policy: b(a|s) as 2D list of shape (num_states, num_actions).
        target_policy: pi(a|s) as 2D list of shape (num_states, num_actions).
        num_states: Number of states.
        num_actions: Number of actions.
        n: Number of steps for the n-step return.
        alpha: Learning rate.
        gamma: Discount factor.

    Returns:
        V: numpy array of shape (num_states,) with estimated state values.
    """
    # Convert policies to numpy arrays for easier indexing
    b = np.array(behavior_policy)
    pi = np.array(target_policy)
    
    # Initialize state-value function to zeros
    V = np.zeros(num_states)
    
    # Process each episode
    for episode in episodes:
        T = len(episode)
        
        # Extract states, actions, rewards for easier access
        states = [episode[k][0] for k in range(T)]
        actions = [episode[k][1] for k in range(T)]
        rewards = [episode[k][2] for k in range(T)]
        
        # For each time step
        for t in range(T):
            # Determine the end of the n-step window
            end_idx = min(t + n, T)
            
            # Compute the importance sampling ratio for the n-step window
            rho = 1.0
            for k in range(t, end_idx):
                s = states[k]
                a = actions[k]
                if b[s, a] > 0:
                    rho *= (pi[s, a] / b[s, a])
                else:
                    rho = 0.0
                    break
            
            # Compute the n-step return
            G = 0.0
            for k in range(t, end_idx):
                G += (gamma ** (k - t)) * rewards[k]
            
            # Add bootstrapping if we haven't reached the end
            if end_idx < T:
                G += (gamma ** n) * V[states[end_idx]]
            
            # Apply the TD update
            V[states[t]] += alpha * rho * (G - V[states[t]])
    
    return V