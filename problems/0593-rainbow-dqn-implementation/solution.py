import numpy as np

def rainbow_dqn_loss(
    online_V_logits: np.ndarray,
    online_A_logits: np.ndarray,
    target_V_logits: np.ndarray,
    target_A_logits: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    dones: np.ndarray,
    gamma_n: float,
    v_min: float,
    v_max: float,
    n_atoms: int
) -> tuple:
    """
    Compute the Rainbow DQN training loss combining dueling architecture,
    double Q-learning, and distributional (categorical) Bellman projection.
    
    Returns:
        Tuple of (losses, priorities, next_actions)
    """
    batch_size = len(actions)
    support = np.linspace(v_min, v_max, n_atoms)
    delta_z = (v_max - v_min) / (n_atoms - 1)
    
    def get_probabilities(V_logits, A_logits):
        # Dueling architecture: Q = V + A - mean(A)
        # Reshape V_logits to (batch, 1, n_atoms) for broadcasting over actions
        V_expanded = np.expand_dims(V_logits, axis=1)
        A_mean = np.mean(A_logits, axis=1, keepdims=True)
        Q_logits = V_expanded + (A_logits - A_mean)
        
        # Softmax over the atom/distribution dimension (axis=-1)
        shift_logits = Q_logits - np.max(Q_logits, axis=-1, keepdims=True)
        exp_logits = np.exp(shift_logits)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        return probs

    # 1. Compute probability distributions for online and target networks
    online_probs = get_probabilities(online_V_logits, online_A_logits)
    target_probs = get_probabilities(target_V_logits, target_A_logits)
    
    # 2. Double Action Selection using the online network's expected values
    # Expected Q-value for each action: sum_z (prob_z * z)
    online_expected_Q = np.sum(online_probs * support, axis=-1)
    next_actions = np.argmax(online_expected_Q, axis=-1)
    
    # 3. Categorical Bellman Projection
    projected_dist = np.zeros((batch_size, n_atoms), dtype=float)
    
    for i in range(batch_size):
        # Extract the target distribution corresponding to the double DQN chosen action
        next_a = next_actions[i]
        p_target = target_probs[i, next_a]
        
        reward = rewards[i]
        done = dones[i]
        
        if done:
            # For terminal states, project all probability mass to the reward clipped to the support bounds
            tz = np.clip(reward, v_min, v_max)
            b = (tz - v_min) / delta_z
            l = int(np.floor(b))
            u = int(np.ceil(b))
            
            if l == u:
                projected_dist[i, l] += 1.0
            else:
                projected_dist[i, l] += (u - b)
                projected_dist[i, u] += (b - l)
        else:
            # For non-terminal states, shift each atom by the reward and scale by discount factor
            for j in range(n_atoms):
                tz = np.clip(reward + gamma_n * support[j], v_min, v_max)
                b = (tz - v_min) / delta_z
                l = int(np.floor(b))
                u = int(np.ceil(b))
                
                if l == u:
                    projected_dist[i, l] += p_target[j]
                else:
                    projected_dist[i, l] += p_target[j] * (u - b)
                    projected_dist[i, u] += p_target[j] * (b - l)
                    
    # 4. Compute Cross-Entropy Loss for the action actually taken
    # Extract the online probability distributions for the executed actions
    online_action_probs = online_probs[np.arange(batch_size), actions]
    
    epsilon = 1e-8
    losses = -np.sum(projected_dist * np.log(online_action_probs + epsilon), axis=-1)
    
    # 5. Compute Prioritized Experience Replay priorities
    priorities = losses + 1e-6
    
    # Round outputs to 4 decimal places
    losses_out = [round(float(l), 4) for l in losses]
    priorities_out = [round(float(p), 4) for p in priorities]
    next_actions_out = [int(a) for a in next_actions]
    
    return losses_out, priorities_out, next_actions_out