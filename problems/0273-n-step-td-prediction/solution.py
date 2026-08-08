import torch

def n_step_td_prediction(
    episodes: list[list[tuple[int, float]]],
    n_states: int,
    n: int,
    gamma: float,
    alpha: float
) -> torch.Tensor:
    """
    Perform n-step TD prediction to estimate state values.
    
    Args:
        episodes: List of episodes. Each episode is a list of (state, reward) tuples.
                 The reward at index i is the reward received AFTER leaving state i.
                 Episodes end with a terminal transition (last state's reward is the final reward).
        n_states: Number of states (states are integers 0 to n_states-1)
        n: Number of steps for n-step TD
        gamma: Discount factor
        alpha: Learning rate
        
    Returns:
        V: Estimated state values as torch.Tensor of shape (n_states,)
    """
    # Initialize state values to zeros
    V = torch.zeros(n_states, dtype=torch.float32)
    
    # Process each episode
    for episode in episodes:
        # Length of the episode (number of transitions)
        T = len(episode)
        
        # Extract states and rewards from episode
        states = [episode[t][0] for t in range(T)]
        rewards = [episode[t][1] for t in range(T)]
        
        # For each time step where we can update
        for t in range(T):
            state = states[t]
            
            # Determine the end of the n-step horizon
            # For n-step TD, we look up to n steps ahead or until the end of episode
            horizon = min(t + n, T)  # t+n steps ahead, but not beyond episode end
            
            # Calculate n-step return
            G = 0.0
            # Sum discounted rewards for n steps (or until episode end)
            for i in range(t, horizon):
                G += (gamma ** (i - t)) * rewards[i]
            
            # If we haven't reached the end of episode, bootstrap from V at horizon
            if horizon < T:
                # Bootstrap from value of state at the horizon
                G += (gamma ** (horizon - t)) * V[states[horizon]]
            # If horizon == T, we're at terminal state, no bootstrap needed
            
            # Update value using TD error
            V[state] += alpha * (G - V[state])
    
    return V