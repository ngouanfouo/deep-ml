import numpy as np

def goal_conditioned_rollout(
    initial_state: np.ndarray,
    actions: np.ndarray,
    goal_state: np.ndarray,
    transition_matrix: np.ndarray,
    action_matrix: np.ndarray,
    goal_matrix: np.ndarray,
    anchor_strength: float = 0.5,
    anchor_power: float = 2.0
) -> dict:
    """
    Roll out a goal-conditioned world model with future anchoring.
    
    Anchor weight at step k (0-indexed) of T total steps:
        alpha_k = anchor_strength * ((k+1) / T) ^ anchor_power
    
    Args:
        initial_state: Shape (D,)
        actions: Shape (T, A) - sequence of T actions
        goal_state: Shape (D,) - desired goal state
        transition_matrix: Shape (D, D) - state transition weights
        action_matrix: Shape (D, A) - action influence weights
        goal_matrix: Shape (D, D) - goal conditioning weights
        anchor_strength: Max anchoring intensity in [0, 1]
        anchor_power: Controls anchoring ramp-up speed
    
    Returns:
        dict with keys:
        - 'trajectory': np.ndarray shape (T+1, D)
        - 'anchor_weights': np.ndarray shape (T,)
        - 'raw_predictions': np.ndarray shape (T, D)
    """
    T = actions.shape[0]
    D = initial_state.shape[0]
    
    # Initialize trajectory with initial state
    trajectory = [initial_state.copy()]
    raw_predictions = []
    anchor_weights = []
    
    current_state = initial_state.copy()
    
    for k in range(T):
        # Compute the goal difference: goal - current_state
        goal_diff = goal_state - current_state
        
        # Compute pre-activation: W_s @ state + W_a @ action + W_g @ goal_diff
        pre_activation = (transition_matrix @ current_state + 
                         action_matrix @ actions[k] + 
                         goal_matrix @ goal_diff)
        
        # Apply tanh activation to get raw prediction
        raw_pred = np.tanh(pre_activation)
        raw_predictions.append(raw_pred.copy())
        
        # Compute anchor weight for this step
        # alpha_k = anchor_strength * ((k+1) / T) ^ anchor_power
        alpha_k = anchor_strength * ((k + 1) / T) ** anchor_power
        
        # Apply future anchoring: blend raw prediction with goal
        # anchored = (1 - alpha) * raw_pred + alpha * goal_state
        anchored = (1 - alpha_k) * raw_pred + alpha_k * goal_state
        
        anchor_weights.append(alpha_k)
        
        # Update current state for next step
        current_state = anchored
        trajectory.append(anchored.copy())
    
    return {
        'trajectory': np.array(trajectory),
        'anchor_weights': np.array(anchor_weights),
        'raw_predictions': np.array(raw_predictions)
    }


def compute_future_anchor_loss(
    trajectory: np.ndarray,
    goal_state: np.ndarray,
    raw_predictions: np.ndarray,
    lambda_goal: float = 1.0,
    lambda_smooth: float = 0.1,
    lambda_consistency: float = 0.1
) -> dict:
    """
    Compute multi-component loss for the goal-conditioned world model.
    
    Args:
        trajectory: Shape (T+1, D)
        goal_state: Shape (D,)
        raw_predictions: Shape (T, D)
        lambda_goal: Weight for goal-reaching loss
        lambda_smooth: Weight for smoothness loss
        lambda_consistency: Weight for consistency loss
    
    Returns:
        dict with 'goal_loss', 'smoothness_loss', 'consistency_loss', 'total_loss'
    """
    T = len(trajectory) - 1
    D = trajectory.shape[1]
    
    # 1. Goal-reaching loss: squared difference between final state and goal
    final_state = trajectory[-1]
    goal_loss = np.sum((final_state - goal_state) ** 2)
    
    # 2. Smoothness loss: mean squared difference between consecutive states
    smoothness_loss = 0.0
    for t in range(T):
        diff = trajectory[t + 1] - trajectory[t]
        smoothness_loss += np.sum(diff ** 2)
    smoothness_loss = smoothness_loss / T  # Average over timesteps
    
    # 3. Consistency loss: mean squared difference between anchored and raw predictions
    consistency_loss = 0.0
    for t in range(T):
        # Anchored prediction is the trajectory state at time t+1
        anchored_pred = trajectory[t + 1]
        raw_pred = raw_predictions[t]
        consistency_loss += np.sum((anchored_pred - raw_pred) ** 2)
    consistency_loss = consistency_loss / T  # Average over timesteps
    
    # Total loss: weighted sum of all components
    total_loss = lambda_goal * goal_loss + lambda_smooth * smoothness_loss + lambda_consistency * consistency_loss
    
    return {
        'goal_loss': float(goal_loss),
        'smoothness_loss': float(smoothness_loss),
        'consistency_loss': float(consistency_loss),
        'total_loss': float(total_loss)
    }