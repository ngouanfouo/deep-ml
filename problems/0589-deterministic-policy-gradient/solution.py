import numpy as np

def deterministic_policy_gradient(
    states: np.ndarray, 
    actions: np.ndarray, 
    rewards: np.ndarray, 
    next_states: np.ndarray, 
    dones: np.ndarray, 
    theta: np.ndarray, 
    w: np.ndarray, 
    gamma: float, 
    alpha_theta: float, 
    alpha_w: float
) -> tuple:
    """
    Perform one update step of the Deterministic Policy Gradient algorithm
    with linear function approximation for both the actor and critic.
    
    Args:
        states: Batch of states, shape (N, state_dim)
        actions: Batch of actions taken, shape (N,)
        rewards: Batch of rewards, shape (N,)
        next_states: Batch of next states, shape (N, state_dim)
        dones: Batch of terminal flags, shape (N,)
        theta: Policy parameters, shape (state_dim,)
        w: Critic parameters, shape (state_dim + 1,)
        gamma: Discount factor
        alpha_theta: Policy learning rate
        alpha_w: Critic learning rate
    
    Returns:
        Tuple of (theta_new, w_new) as lists rounded to 4 decimal places
    """
    N, state_dim = states.shape
    
    # Extract weights corresponding to state features and the action feature
    w_state = w[:state_dim]
    w_action = w[state_dim]
    
    # -------------------------------------------------------------------------
    # 1. Critic Update (w)
    # -------------------------------------------------------------------------
    # Compute current Q-values for the transitions: Q(s, a)
    q_current = np.dot(states, w_state) + actions * w_action
    
    # Predict the next actions using the deterministic target/current policy
    next_actions = np.dot(next_states, theta)
    
    # Compute next Q-values: Q(s', a')
    q_next = np.dot(next_states, w_state) + next_actions * w_action
    
    # Compute TD targets and temporal difference errors
    targets = rewards + gamma * (1.0 - dones) * q_next
    td_errors = targets - q_current
    
    # Construct the state-action feature matrix phi = [s, a]
    critic_features = np.hstack((states, actions.reshape(-1, 1)))
    
    # Compute the mean semi-gradient over the batch
    critic_gradient = np.mean(td_errors.reshape(-1, 1) * critic_features, axis=0)
    w_new = w + alpha_w * critic_gradient
    
    # -------------------------------------------------------------------------
    # 2. Actor Update (theta)
    # -------------------------------------------------------------------------
    # dQ/da = w_action; d_mu/d_theta = s
    # The deterministic policy gradient is dQ/da * d_mu/d_theta
    actor_gradients = w_action * states
    mean_actor_gradient = np.mean(actor_gradients, axis=0)
    
    # Gradient ascent step for maximizing expected return
    theta_new = theta + alpha_theta * mean_actor_gradient
    
    # -------------------------------------------------------------------------
    # Format and Output
    # -------------------------------------------------------------------------
    theta_out = [round(float(t), 4) for t in theta_new]
    w_out = [round(float(weight), 4) for weight in w_new]
    
    return theta_out, w_out