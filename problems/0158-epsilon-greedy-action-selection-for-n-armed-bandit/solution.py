import numpy as np

def epsilon_greedy(Q, epsilon=0.1):
    """
    Selects an action using epsilon-greedy policy.
    Q: np.ndarray of shape (n,) -- estimated action values
    epsilon: float in [0, 1]
    Returns: int, selected action index
    """
    # Your code here
    n_actions = len(Q)
    
    # With probability epsilon, explore (choose random action)
    if np.random.random() < epsilon:
        return np.random.randint(n_actions)
    else:
        # With probability 1-epsilon, exploit (choose best action)
        # Note: In case of ties, np.argmax returns the first occurrence
        return int(np.argmax(Q))