import numpy as np

def dqn_training_step(
    weights: dict,
    target_weights: dict,
    batch: dict,
    gamma: float,
    learning_rate: float
) -> tuple:
    """
    Perform one DQN training step with backpropagation.
    
    Network: input -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> Q-values
    
    Args:
        weights: dict with 'W1' (state_dim, hidden_dim), 'b1' (hidden_dim,),
                 'W2' (hidden_dim, num_actions), 'b2' (num_actions,)
        target_weights: same structure as weights, for the target network
        batch: dict with:
            'states': (batch_size, state_dim)
            'actions': (batch_size,) int array
            'rewards': (batch_size,) float array
            'next_states': (batch_size, state_dim)
            'dones': (batch_size,) bool array
        gamma: discount factor
        learning_rate: learning rate for SGD
    
    Returns:
        tuple of (updated_weights: dict, loss: float rounded to 4 decimals)
    """
    # Extract batch data
    states = np.array(batch['states'])
    actions = np.array(batch['actions'])
    rewards = np.array(batch['rewards'])
    next_states = np.array(batch['next_states'])
    dones = np.array(batch['dones'])
    
    batch_size = states.shape[0]
    state_dim = states.shape[1]
    hidden_dim = weights['W1'].shape[1]
    num_actions = weights['W2'].shape[1]
    
    # ----- Forward pass through Q-network -----
    # z1 = states @ W1 + b1
    z1 = states @ weights['W1'] + weights['b1']  # (batch_size, hidden_dim)
    
    # h1 = ReLU(z1)
    h1 = np.maximum(0, z1)  # (batch_size, hidden_dim)
    
    # q = h1 @ W2 + b2
    q_values = h1 @ weights['W2'] + weights['b2']  # (batch_size, num_actions)
    
    # Extract Q-values for the actions taken
    q_taken = q_values[np.arange(batch_size), actions]  # (batch_size,)
    
    # ----- Compute TD targets using target network -----
    # Forward pass through target network for next states
    z1_target = next_states @ target_weights['W1'] + target_weights['b1']
    h1_target = np.maximum(0, z1_target)
    q_next = h1_target @ target_weights['W2'] + target_weights['b2']  # (batch_size, num_actions)
    
    # Max Q-value for next states (use max, not argmax)
    max_q_next = np.max(q_next, axis=1)  # (batch_size,)
    
    # TD targets: r + gamma * max_q_next for non-terminal, r for terminal
    td_targets = rewards + gamma * max_q_next * (~dones)  # (batch_size,)
    
    # ----- Compute MSE loss -----
    # Loss = mean((q_taken - td_targets)^2)
    td_errors = q_taken - td_targets
    loss = np.mean(td_errors ** 2)
    
    # ----- Backpropagation -----
    # Gradient of loss w.r.t. q_taken
    d_loss_d_q = 2 * td_errors / batch_size  # (batch_size,)
    
    # Gradient w.r.t. Q-values (only the taken actions get gradients)
    d_loss_d_q_values = np.zeros_like(q_values)  # (batch_size, num_actions)
    d_loss_d_q_values[np.arange(batch_size), actions] = d_loss_d_q
    
    # Gradient w.r.t. W2 and b2
    # d_loss/d_W2 = h1^T @ d_loss_d_q_values
    d_W2 = h1.T @ d_loss_d_q_values  # (hidden_dim, num_actions)
    d_b2 = np.sum(d_loss_d_q_values, axis=0)  # (num_actions,)
    
    # Gradient w.r.t. h1
    d_loss_d_h1 = d_loss_d_q_values @ weights['W2'].T  # (batch_size, hidden_dim)
    
    # Gradient through ReLU (gradient is 1 for positive z1, 0 otherwise)
    d_loss_d_z1 = d_loss_d_h1 * (z1 > 0)  # (batch_size, hidden_dim)
    
    # Gradient w.r.t. W1 and b1
    d_W1 = states.T @ d_loss_d_z1  # (state_dim, hidden_dim)
    d_b1 = np.sum(d_loss_d_z1, axis=0)  # (hidden_dim,)
    
    # ----- Update weights using SGD -----
    updated_weights = {
        'W1': weights['W1'] - learning_rate * d_W1,
        'b1': weights['b1'] - learning_rate * d_b1,
        'W2': weights['W2'] - learning_rate * d_W2,
        'b2': weights['b2'] - learning_rate * d_b2
    }
    
    # Round loss to 4 decimal places
    loss_rounded = round(float(loss), 4)
    
    return updated_weights, loss_rounded