import numpy as np

def async_ppo_pipeline(worker_batches: list, theta_init: list, alpha: float, gamma_stale: float, clip_eps: float, max_staleness: int) -> dict:
    """
    Simulate asynchronous PPO training pipeline.
    
    Args:
        worker_batches: list of dicts with 'policy_version', 'states', 'actions', 'old_log_probs', 'advantages'
        theta_init: initial policy parameters (n_features x n_actions)
        alpha: base learning rate
        gamma_stale: staleness decay factor for learning rate
        clip_eps: PPO clipping epsilon
        max_staleness: maximum allowed policy lag
    
    Returns:
        dict with 'theta', 'batches_used', 'batches_discarded', 'avg_clip_fraction'
    """
    # Convert theta_init to numpy array
    theta = np.array(theta_init, dtype=float)
    n_features, n_actions = theta.shape
    
    # Initialize tracking variables
    learner_version = 0
    batches_used = 0
    batches_discarded = 0
    clip_fractions = []
    
    # Softmax function
    def softmax(logits):
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    
    # Compute policy probabilities and log probabilities for given states
    def policy_probs_and_logprobs(states, theta):
        states = np.array(states)
        if len(states.shape) == 1:
            states = states.reshape(1, -1)
        logits = states @ theta
        probs = softmax(logits)
        log_probs = np.log(probs + 1e-10)
        return probs, log_probs
    
    # Compute score function for linear softmax policy
    def compute_score(states, actions, theta):
        states = np.array(states)
        if len(states.shape) == 1:
            states = states.reshape(1, -1)
        probs, _ = policy_probs_and_logprobs(states, theta)
        
        scores = []
        for i, (state, action) in enumerate(zip(states, actions)):
            e_a = np.zeros(n_actions)
            e_a[action] = 1.0
            grad_logit = e_a - probs[i]
            score = np.outer(state, grad_logit)
            scores.append(score)
        return np.array(scores)
    
    # Process each batch sequentially
    for batch in worker_batches:
        policy_version = batch['policy_version']
        states = batch['states']
        actions = batch['actions']
        old_log_probs = np.array(batch['old_log_probs'])
        advantages = np.array(batch['advantages'])
        
        # Compute policy lag
        lag = learner_version - policy_version
        
        # Check staleness
        if lag > max_staleness:
            batches_discarded += 1
            continue
        
        # Compute new log probabilities under current theta
        _, new_log_probs_all = policy_probs_and_logprobs(states, theta)
        # Extract only the log probabilities for the actions taken
        new_log_probs = np.array([new_log_probs_all[i][actions[i]] for i in range(len(actions))])
        
        # Compute importance sampling ratios
        ratios = np.exp(new_log_probs - old_log_probs)
        
        # Compute PPO objective terms
        surr1 = ratios * advantages
        surr2 = np.clip(ratios, 1 - clip_eps, 1 + clip_eps) * advantages
        
        # Determine clipped samples (surr1 > surr2 means the ratio was clipped)
        clipped = surr1 > surr2
        clip_fraction = np.mean(clipped)
        clip_fractions.append(clip_fraction)
        
        # Only use non-clipped samples for gradient
        non_clipped_mask = ~clipped
        non_clipped_indices = np.where(non_clipped_mask)[0]
        
        if len(non_clipped_indices) > 0:
            # Compute scores for non-clipped samples under current theta
            non_clipped_states = [states[i] for i in non_clipped_indices]
            non_clipped_actions = [actions[i] for i in non_clipped_indices]
            non_clipped_ratios = ratios[non_clipped_indices]
            non_clipped_advantages = advantages[non_clipped_indices]
            
            scores = compute_score(non_clipped_states, non_clipped_actions, theta)
            
            # Accumulate gradient: ratio * advantage * score
            # Average over the FULL batch size
            grad_theta = np.zeros_like(theta)
            for i, idx in enumerate(non_clipped_indices):
                grad_theta += non_clipped_ratios[i] * non_clipped_advantages[i] * scores[i]
            
            # Average over total batch size (not just non-clipped)
            grad_theta /= len(actions)
        else:
            grad_theta = np.zeros_like(theta)
        
        # Apply effective learning rate with staleness decay
        effective_lr = alpha * (gamma_stale ** lag)
        
        # Update theta
        theta += effective_lr * grad_theta
        
        # Increment learner version
        learner_version += 1
        batches_used += 1
    
    # Compute average clip fraction
    if clip_fractions:
        avg_clip_fraction = np.mean(clip_fractions)
    else:
        avg_clip_fraction = 0.0
    
    # Round all values to 4 decimal places
    theta_rounded = np.round(theta, 4).tolist()
    
    return {
        'theta': theta_rounded,
        'batches_used': batches_used,
        'batches_discarded': batches_discarded,
        'avg_clip_fraction': float(np.round(avg_clip_fraction, 4))
    }