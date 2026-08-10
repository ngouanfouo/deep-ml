import numpy as np

def blocking_maze_dyna_q(rows: int, cols: int, initial_walls: list, added_walls: list,
                         start: list, goal: list, change_step: int,
                         gamma: float, alpha: float, epsilon: float,
                         n_planning: int, n_steps: int, seed: int) -> dict:
    """
    Simulate a Dyna-Q agent in a blocking maze where walls change mid-simulation.
    
    Args:
        rows: Number of rows in the grid
        cols: Number of columns in the grid
        initial_walls: List of [row, col] wall positions at start
        added_walls: List of [row, col] walls added at change_step
        start: [row, col] start position
        goal: [row, col] goal position
        change_step: Timestep at which added_walls appear
        gamma: Discount factor
        alpha: Learning rate
        epsilon: Exploration probability
        n_planning: Number of planning updates per real step
        n_steps: Total number of simulation steps
        seed: Random seed
    
    Returns:
        Dictionary with episodes_completed, cumulative_reward, 
        model_entries, and stale_entries
    """
    # Set random seed
    np.random.seed(seed)
    
    # Convert positions to state indices
    start_state = start[0] * cols + start[1]
    goal_state = goal[0] * cols + goal[1]
    
    # Initialize walls as a set for O(1) lookup
    current_walls = set()
    for wall in initial_walls:
        current_walls.add(wall[0] * cols + wall[1])
    
    # Store added walls for later
    added_walls_set = set()
    for wall in added_walls:
        added_walls_set.add(wall[0] * cols + wall[1])
    
    # Initialize Q-table: (rows * cols) x 4 actions
    # Actions: 0=up, 1=down, 2=left, 3=right
    n_states = rows * cols
    n_actions = 4
    Q = np.zeros((n_states, n_actions))
    
    # Initialize model: dict mapping (state, action) -> (next_state, reward)
    model = {}
    
    # Track stale entries
    stale_entries = 0
    
    # Simulation variables
    state = start_state
    episodes_completed = 0
    cumulative_reward = 0.0
    
    # Helper function to check if a position is a wall
    def is_wall(pos):
        return pos in current_walls
    
    # Helper function to get next state and reward given state and action
    def take_action(s, a):
        row = s // cols
        col = s % cols
        
        # Compute next position
        if a == 0:  # up
            next_row, next_col = row - 1, col
        elif a == 1:  # down
            next_row, next_col = row + 1, col
        elif a == 2:  # left
            next_row, next_col = row, col - 1
        else:  # right (action 3)
            next_row, next_col = row, col + 1
        
        # Check if move is valid (within bounds and not a wall)
        if (0 <= next_row < rows and 0 <= next_col < cols and 
            not is_wall(next_row * cols + next_col)):
            next_s = next_row * cols + next_col
        else:
            next_s = s
        
        # Check if reached goal
        reward = 1.0 if next_s == goal_state else 0.0
        
        return next_s, reward
    
    # Main simulation loop
    for step in range(n_steps):
        # Check if we need to add walls at this step
        if step == change_step:
            # Add the new walls
            for wall_pos in added_walls_set:
                current_walls.add(wall_pos)
            
            # Check for stale model entries
            # A model entry is stale if the stored (next_state, reward) no longer
            # matches what would happen in the current environment
            stale_entries = 0
            for (s, a), (next_s, reward) in model.items():
                # Check if this (s, a) pair is still valid and leads to the same outcome
                if not is_wall(s):  # Only check if state is not a wall
                    actual_next_s, actual_reward = take_action(s, a)
                    if actual_next_s != next_s or abs(actual_reward - reward) > 1e-6:
                        stale_entries += 1
        
        # 1. Select action using epsilon-greedy
        if np.random.random() < epsilon:
            # Explore: choose random action
            action = np.random.randint(n_actions)
        else:
            # Exploit: choose action with max Q-value (numpy argmax handles ties)
            action = np.argmax(Q[state])
        
        # 2. Take action in real environment
        next_state, reward = take_action(state, action)
        
        # 3. Update Q-value using Q-learning update rule
        # Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
        best_next = np.max(Q[next_state]) if next_state != goal_state else 0.0
        td_target = reward + gamma * best_next
        Q[state, action] += alpha * (td_target - Q[state, action])
        
        # 4. Record transition in model
        model[(state, action)] = (next_state, reward)
        
        # 5. Perform n_planning planning steps
        for _ in range(n_planning):
            if len(model) > 0:
                # Sample uniformly from previously seen (state, action) pairs
                idx = np.random.randint(len(model))
                sampled_s, sampled_a = list(model.keys())[idx]
                sampled_next_s, sampled_reward = model[(sampled_s, sampled_a)]
                
                # Apply Q-learning update using model predictions
                best_next_sampled = np.max(Q[sampled_next_s]) if sampled_next_s != goal_state else 0.0
                td_target_sampled = sampled_reward + gamma * best_next_sampled
                Q[sampled_s, sampled_a] += alpha * (td_target_sampled - Q[sampled_s, sampled_a])
        
        # Update cumulative reward
        cumulative_reward += reward
        
        # Move to next state
        state = next_state
        
        # Check if reached goal
        if state == goal_state:
            episodes_completed += 1
            state = start_state  # Reset to start
    
    # Compute final model_entries
    model_entries = len(model)
    
    # Recalculate stale_entries one more time after all steps
    # (in case we need to ensure it's accurate at the end)
    stale_entries = 0
    for (s, a), (next_s, reward) in model.items():
        if not is_wall(s):  # Only check if state is not a wall
            actual_next_s, actual_reward = take_action(s, a)
            if actual_next_s != next_s or abs(actual_reward - reward) > 1e-6:
                stale_entries += 1
    
    return {
        "episodes_completed": episodes_completed,
        "cumulative_reward": round(float(cumulative_reward), 4),
        "model_entries": model_entries,
        "stale_entries": stale_entries
    }