import numpy as np

def td_lambda_forward_view(episodes: list, n_states: int, gamma: float, alpha: float, lam: float) -> list:
    """
    Implement forward-view TD(lambda) prediction.
    
    Args:
        episodes: List of episodes, each a list of (state, reward, next_state, done) tuples
        n_states: Number of states
        gamma: Discount factor
        alpha: Learning rate
        lam: Lambda parameter for trace decay
    
    Returns:
        List of estimated state values rounded to 4 decimal places
    """
    # Initialize value function
    V = np.zeros(n_states, dtype=np.float64)
    
    for episode in episodes:
        # Freeze the value function at the start of the episode
        V_snapshot = V.copy()
        H = len(episode)
        
        # Array to store the lambda-returns for the current episode
        G_lambdas = np.zeros(H, dtype=np.float64)
        
        # Compute the lambda-return for each time step t
        for t in range(H):
            n_step_returns = []
            
            # Generate all possible n-step returns from step t up to the end of the episode
            for n in range(1, H - t + 1):
                g_n = 0.0
                # Accumulate rewards for the n steps
                for i in range(n):
                    curr_step = episode[t + i]
                    g_n += (gamma ** i) * curr_step[1] # curr_step[1] is the reward
                
                # If the last transition in this n-step sequence is not terminal, bootstrap
                last_step = episode[t + n - 1]
                if not last_step[3]: # last_step[3] is the done flag
                    g_n += (gamma ** n) * V_snapshot[last_step[2]] # last_step[2] is next_state
                    
                n_step_returns.append(g_n)
            
            # Combine the n-step returns using lambda weights
            g_lambda = 0.0
            num_returns = len(n_step_returns)
            
            for idx, g_n in enumerate(n_step_returns):
                n = idx + 1
                if n < num_returns:
                    # Shorter returns get the decaying weight
                    weight = (1.0 - lam) * (lam ** (n - 1))
                else:
                    # The longest remaining return gets all the residual weight
                    weight = lam ** (n - 1)
                
                g_lambda += weight * g_n
                
            G_lambdas[t] = g_lambda
            
        # Sequentially apply the semi-gradient updates using the snapshot values in the error term
        for t in range(H):
            state = episode[t][0]
            V[state] += alpha * (G_lambdas[t] - V_snapshot[state])
            
    # Return as a list rounded to 4 decimal places
    return [round(float(val), 4) for val in V]