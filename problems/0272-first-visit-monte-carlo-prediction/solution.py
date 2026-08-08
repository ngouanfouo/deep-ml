import torch

def first_visit_mc_prediction(
    episodes: list[list[tuple[int, float]]],
    n_states: int,
    gamma: float
) -> torch.Tensor:
    """
    First-Visit Monte Carlo prediction implementation.
    
    This version uses a more explicit episode-by-episode approach with
    pre-computed returns for clarity.
    """
    # Initialize value estimates
    V = torch.zeros(n_states, dtype=torch.float32)
    
    # Initialize dictionary to store returns for each state
    returns_dict = {state: [] for state in range(n_states)}
    
    # Process each episode
    for episode in episodes:
        # Track first visit occurrences
        first_visits = {}
        
        # Find first visit time for each state
        for t, (state, _) in enumerate(episode):
            if state not in first_visits:
                first_visits[state] = t
        
        # Calculate returns for each first-visited state
        for state, first_visit_time in first_visits.items():
            # Calculate return from first visit time to end
            G = 0.0
            for t in range(first_visit_time, len(episode)):
                G += (gamma ** (t - first_visit_time)) * episode[t][1]
            returns_dict[state].append(G)
    
    # Average returns for each state
    for state in range(n_states):
        if returns_dict[state]:
            V[state] = torch.mean(torch.tensor(returns_dict[state], dtype=torch.float32))
    
    return V