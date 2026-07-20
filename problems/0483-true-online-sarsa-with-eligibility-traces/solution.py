import numpy as np

def true_online_sarsa_lambda(episodes, n_features, alpha, gamma, lam):
    """
    Implement True Online SARSA(lambda) with linear function approximation.

    Args:
        episodes: list of episodes. Each episode is a list of tuples
                  (x, reward, x_next, done) where x and x_next are
                  numpy arrays of shape (n_features...).
        n_features: int, dimension of feature vectors
        alpha: float, step size
        gamma: float, discount factor
        lam: float, trace decay parameter

    Returns:
        list of floats: learned weight vector rounded to 5 decimal places
    """
    w = np.zeros(n_features)
    
    for episode in episodes:
        z = np.zeros(n_features)
        Q_old = 0.0
        
        for x, reward, x_next, done in episode:
            Q = np.dot(w, x)
            Q_next = 0.0 if done else np.dot(w, x_next)
            
            delta = reward + gamma * Q_next - Q
            
            # Update eligibility trace (Dutch trace)
            z = gamma * lam * z + (1.0 - alpha * gamma * lam * np.dot(z, x)) * x
            
            # Update weights
            w = w + alpha * (delta + Q - Q_old) * z - alpha * (Q - Q_old) * x
            
            Q_old = Q_next
            
    return [round(float(val), 5) for val in w]