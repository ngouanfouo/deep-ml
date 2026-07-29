import numpy as np

def natural_policy_gradient_step(
    theta: np.ndarray,
    trajectories: list,
    feature_vectors: dict,
    actions: list,
    gamma: float,
    alpha: float,
    epsilon: float = 1e-4
) -> dict:
    """
    Perform a single Natural Policy Gradient update step.

    Args:
        theta: (d,) parameter vector for the softmax policy
        trajectories: list of episodes; each episode is a list of (state, action, reward) tuples
        feature_vectors: dict mapping (state, action) -> list of floats (feature vector)
        actions: list of all possible actions
        gamma: discount factor
        alpha: learning rate
        epsilon: regularization constant for FIM inversion

    Returns:
        Dictionary with keys:
          'vanilla_gradient': np.ndarray of shape (d,)
          'fisher_matrix': np.ndarray of shape (d, d)
          'natural_gradient': np.ndarray of shape (d,)
          'updated_theta': np.ndarray of shape (d,)
    """
    d = len(theta)
    
    # Pre-convert feature vectors to numpy arrays for efficiency
    phi = {k: np.array(v, dtype=float) for k, v in feature_vectors.items()}
    
    total_episodes = len(trajectories)
    episode_gradients = []
    all_scores = []
    
    for episode in trajectories:
        T = len(episode)
        
        # 1. Compute discounted returns G_t for each step in the episode
        returns = np.zeros(T)
        G = 0.0
        for t in reversed(range(T)):
            state, action, reward = episode[t]
            G = reward + gamma * G
            returns[t] = G
            
        # 2. Compute the policy score (log gradient) for each step
        episode_gradient = np.zeros(d)
        for t in range(T):
            state, action, reward = episode[t]
            
            # Compute softmax probabilities for all actions in the current state
            action_probs = {}
            scores = []
            for a in actions:
                score = np.dot(theta, phi[(state, a)])
                scores.append(score)
            
            # Numeric stability trick for softmax
            scores = np.array(scores)
            max_score = np.max(scores)
            exp_scores = np.exp(scores - max_score)
            sum_exp_scores = np.sum(exp_scores)
            
            for idx, a in enumerate(actions):
                action_probs[a] = exp_scores[idx] / sum_exp_scores
                
            # Expected feature vector under the current policy distribution
            expected_phi = np.zeros(d)
            for a in actions:
                expected_phi += action_probs[a] * phi[(state, a)]
                
            # Score vector: grad_theta log pi(a_t | s_t)
            score_t = phi[(state, action)] - expected_phi
            all_scores.append(score_t)
            
            # Accumulate the vanilla policy gradient step contribution
            episode_gradient += score_t * returns[t]
            
        episode_gradients.append(episode_gradient)
        
    # Average vanilla gradient across all trajectories
    vanilla_gradient = np.mean(episode_gradients, axis=0)
    
    # Calculate empirical Fisher Information Matrix (FIM)
    # Average of the outer products of score vectors over all experienced state-action pairs
    fisher_matrix = np.zeros((d, d))
    if all_scores:
        for score in all_scores:
            fisher_matrix += np.outer(score, score)
        fisher_matrix /= len(all_scores)
        
    # Regularize the FIM for numerical stability
    regularized_fim = fisher_matrix + epsilon * np.eye(d)
    
    # Solve for the natural gradient direction: (F + epsilon * I) * ng = vanilla_gradient
    natural_gradient = np.linalg.solve(regularized_fim, vanilla_gradient)
    
    # Compute the parameter update
    updated_theta = theta + alpha * natural_gradient
    
    return {
        'vanilla_gradient': vanilla_gradient,
        'fisher_matrix': fisher_matrix,
        'natural_gradient': natural_gradient,
        'updated_theta': updated_theta
    }