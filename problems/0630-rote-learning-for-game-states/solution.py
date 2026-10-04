from collections import defaultdict

def rote_learning(episodes: list, gamma: float) -> dict:
    """
    Build a rote-learning value table from game episodes.
    
    Args:
        episodes: List of episodes. Each episode is a list of (state, reward) tuples.
        gamma: Discount factor.
    
    Returns:
        Dictionary mapping each state to its average discounted return,
        rounded to 4 decimal places.
    """
    returns = defaultdict(list)
    
    for episode in episodes:
        G = 0.0
        # Work backwards through the episode
        for state, reward in reversed(episode):
            G = reward + gamma * G
            returns[state].append(G)
    
    # Average the observed returns for each state
    result = {}
    for state, values in returns.items():
        avg = sum(values) / len(values)
        result[state] = round(avg, 4)
    
    return result