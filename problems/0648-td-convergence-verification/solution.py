import numpy as np

def td_lambda_convergence(
    episodes: list,
    n_states: int,
    gamma: float,
    lam: float,
    alpha: float,
    true_values: np.ndarray,
    tolerance: float
) -> tuple:
    """
    Run TD(lambda) with eligibility traces and monitor convergence.
    
    Args:
        episodes: List of episodes, each a list of (state, reward, next_state, done) tuples
        n_states: Number of states
        gamma: Discount factor
        lam: Trace decay parameter (lambda)
        alpha: Learning rate
        true_values: True value function for RMSE computation
        tolerance: Convergence threshold on RMSE
    
    Returns:
        Tuple of (v_final, rmse_history, convergence_episode)
    """
    # Initialize value estimates to zero
    V = np.zeros(n_states)
    
    # Store RMSE history
    rmse_history = []
    convergence_episode = -1
    
    # Process each episode
    for episode_idx, episode in enumerate(episodes):
        # Initialize eligibility trace for this episode
        E = np.zeros(n_states)
        
        # Process each transition in the episode
        for state, reward, next_state, done in episode:
            # Compute TD error
            # For terminal transitions, bootstrap value is 0
            if done:
                V_next = 0
            else:
                V_next = V[next_state]
            
            delta = reward + gamma * V_next - V[state]
            
            # Decay eligibility trace (accumulating traces)
            E *= gamma * lam
            E[state] += 1
            
            # Update all value estimates
            V += alpha * delta * E
        
        # Compute RMSE after this episode
        mse = np.mean((V - true_values) ** 2)
        rmse = np.sqrt(mse)
        rmse_history.append(float(rmse))  # Convert to Python float
        
        # Check for convergence
        if rmse < tolerance and convergence_episode == -1:
            convergence_episode = episode_idx
    
    # Round final values to 4 decimal places and convert to Python floats
    v_final = [float(round(v, 4)) for v in V]
    
    # Round RMSE history to 4 decimal places and convert to Python floats
    rmse_history = [float(round(rmse, 4)) for rmse in rmse_history]
    
    return (v_final, rmse_history, convergence_episode)