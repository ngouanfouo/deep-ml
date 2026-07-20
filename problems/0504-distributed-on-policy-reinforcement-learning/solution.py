import numpy as np

def distributed_onpolicy_step(worker_data: list, gamma: float, lam: float, clip_eps: float) -> dict:
    """
    Simulate one optimization step of distributed on-policy RL.
    
    Args:
        worker_data: List of dicts with keys 'rewards', 'values', 'dones',
                     'old_log_probs', 'new_log_probs' per worker.
        gamma: Discount factor.
        lam: GAE lambda parameter.
        clip_eps: PPO-style clipping epsilon.
    
    Returns:
        Dictionary with aggregated training statistics.
    """
    all_advantages = []
    all_returns = []
    all_old_values = []
    all_old_log_probs = []
    all_new_log_probs = []
    
    per_worker_mean_returns = []
    total_samples = 0
    
    for worker in worker_data:
        rewards = np.array(worker['rewards'], dtype=np.float64)
        values = np.array(worker['values'], dtype=np.float64)
        dones = np.array(worker['dones'], dtype=np.float64)
        old_log_probs = np.array(worker['old_log_probs'], dtype=np.float64)
        new_log_probs = np.array(worker['new_log_probs'], dtype=np.float64)
        
        T = len(rewards)
        total_samples += T
        
        # 1. Compute Generalized Advantage Estimation (GAE)
        advantages = np.zeros(T, dtype=np.float64)
        gae = 0.0
        
        for t in reversed(range(T)):
            # Temporal Difference Error: delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
            next_value = values[t + 1]
            delta = rewards[t] + gamma * next_value * (1.0 - dones[t]) - values[t]
            
            # GAE trace accumulation
            gae = delta + gamma * lam * (1.0 - dones[t]) * gae
            advantages[t] = gae
            
        # Returns are target values computed as: Q(s, a) = Advantage(s, a) + Value(s)
        returns = advantages + values[:T]
        
        # Save worker-specific metric
        per_worker_mean_returns.append(float(np.mean(returns)))
        
        # Accumulate to pool across all distributed workers
        all_advantages.extend(advantages)
        all_returns.extend(returns)
        all_old_values.extend(values[:T])
        all_old_log_probs.extend(old_log_probs)
        all_new_log_probs.extend(new_log_probs)
        
    # Convert pooled elements to numpy arrays
    advantages_arr = np.array(all_advantages, dtype=np.float64)
    returns_arr = np.array(all_returns, dtype=np.float64)
    old_values_arr = np.array(all_old_values, dtype=np.float64)
    old_log_probs_arr = np.array(all_old_log_probs, dtype=np.float64)
    new_log_probs_arr = np.array(all_new_log_probs, dtype=np.float64)
    
    # Calculate baseline raw metrics
    mean_adv = float(np.mean(advantages_arr))
    std_adv = float(np.std(advantages_arr))
    
    # 2. Pooled Advantage Normalization
    if std_adv < 1e-8:
        norm_advantages = advantages_arr - mean_adv
    else:
        norm_advantages = (advantages_arr - mean_adv) / std_adv
        
    # 3. Clipped Surrogate Policy Loss Computation
    # r_t(theta) = exp(log_prob_new - log_prob_old)
    ratios = np.exp(new_log_probs_arr - old_log_probs_arr)
    
    surr1 = ratios * norm_advantages
    surr2 = np.clip(ratios, 1.0 - clip_eps, 1.0 + clip_eps) * norm_advantages
    policy_loss = -np.mean(np.minimum(surr1, surr2))
    
    # 4. Value Loss (MSE between target returns and original state values)
    value_loss = np.mean((returns_arr - old_values_arr) ** 2)
    
    # 5. Compute the fraction of samples where ratios were clipped
    # Ratio is clipped if it violates the strict boundaries [1 - eps, 1 + eps]
    is_clipped = (ratios < (1.0 - clip_eps - 1e-9)) | (ratios > (1.0 + clip_eps + 1e-9))
    clip_fraction = float(np.mean(is_clipped))
    
    return {
        'policy_loss': round(float(policy_loss), 4),
        'value_loss': round(float(value_loss), 4),
        'clip_fraction': round(clip_fraction, 4),
        'mean_advantage_raw': round(mean_adv, 4),
        'std_advantage_raw': round(std_adv, 4),
        'num_samples': int(total_samples),
        'per_worker_mean_return': [round(val, 4) for val in per_worker_mean_returns]
    }