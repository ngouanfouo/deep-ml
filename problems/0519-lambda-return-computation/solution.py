import numpy as np

def compute_lambda_returns(rewards: np.ndarray, values: np.ndarray, gamma: float, lam: float, dones: np.ndarray = None) -> np.ndarray:
    """
    Compute the lambda-return for each timestep in a trajectory.
    
    Args:
        rewards: Rewards at each step, shape (T,)
        values: Value estimates at each state, shape (T+1,)
        gamma: Discount factor in [0, 1]
        lam: Lambda (trace-decay) parameter in [0, 1]
        dones: Optional binary episode termination flags, shape (T,)
    
    Returns:
        Lambda-returns for each timestep, shape (T,)
    """
    T = len(rewards)
    
    if dones is None:
        dones = np.zeros(T, dtype=bool)
    else:
        dones = dones.astype(bool)
        
    lambda_returns = np.zeros(T, dtype=np.float64)
    
    # Initialize the future lambda return target with the final state value estimate V(s_T)
    next_lambda_return = values[T]
    
    # Iterate backwards through the trajectory
    for t in reversed(range(T)):
        if dones[t]:
            # If the episode ends, there is no value bootstrap or future continuation
            lambda_returns[t] = rewards[t]
        else:
            # At the boundary t = T-1, if not done, this correctly evaluates to:
            # rewards[t] + gamma * ((1 - lam) * values[t+1] + lam * values[t+1]) = rewards[t] + gamma * values[t+1]
            lambda_returns[t] = rewards[t] + gamma * ((1 - lam) * values[t + 1] + lam * next_lambda_return)
            
        # Update the tracker for the next iteration step
        next_lambda_return = lambda_returns[t]
        
    return lambda_returns