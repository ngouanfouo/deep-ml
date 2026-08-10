import numpy as np

def shortcut_maze_exploration(
    grid_rows: int,
    grid_cols: int,
    walls_phase1: list,
    walls_phase2: list,
    change_step: int,
    start: list,
    goal: tuple,
    total_steps: int,
    alpha: float,
    gamma: float,
    epsilon: float,
    kappa: float,
    n_planning: int,
    seed: int
) -> dict:
    """
    Simulate a Dyna-Q+ agent in a changing maze environment.
    
    Args:
        grid_rows: Number of rows in the grid
        grid_cols: Number of columns in the grid
        walls_phase1: List of [r,c] blocked cells before change_step
        walls_phase2: List of [r,c] blocked cells after change_step
        change_step: Step at which maze changes
        start: [r, c] starting position
        goal: (r, c) goal position
        total_steps: Total number of timesteps
        alpha: Learning rate
        gamma: Discount factor
        epsilon: Exploration rate
        kappa: Exploration bonus coefficient
        n_planning: Number of planning updates per step
        seed: Random seed
    
    Returns:
        dict with 'episodes_completed', 'total_reward', 'cumulative_reward'
    """
    # Set random seed
    rng = np.random.RandomState(seed)
    
    # Convert positions to state indices
    start_state = start[0] * grid_cols + start[1]
    goal_state = goal[0] * grid_cols + goal[1]
    
    # Initialize walls as sets for O(1) lookup
    walls_phase1_set = set()
    for wall in walls_phase1:
        walls_phase1_set.add(wall[0] * grid_cols + wall[1])
    
    walls_phase2_set = set()
    for wall in walls_phase2:
        walls_phase2_set.add(wall[0] * grid_cols + wall[1])
    
    # Current walls - start with phase 1
    current_walls = walls_phase1_set.copy()
    
    # Initialize Q-table: (rows * cols) x 4 actions
    # Actions: 0=up, 1=down, 2=left, 3=right
    n_states = grid_rows * grid_cols
    n_actions = 4
    Q = np.zeros((n_states, n_actions))
    
    # Model: dict mapping (state, action) -> (next_state, reward)
    model = {}
    
    # Last visit timestep for each (state, action) pair
    last_visit = {}
    
    # Simulation variables
    state = start_state
    episodes_completed = 0
    total_reward = 0.0
    cumulative_reward = [0.0]  # Will be filled with cumulative reward after each step
    
    # Helper function to check if a position is a wall
    def is_wall(pos):
        return pos in current_walls
    
    # Helper function to get next state and reward given state and action
    def take_action(s, a):
        row = s // grid_cols
        col = s % grid_cols
        
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
        if (0 <= next_row < grid_rows and 0 <= next_col < grid_cols and 
            not is_wall(next_row * grid_cols + next_col)):
            next_s = next_row * grid_cols + next_col
        else:
            next_s = s
        
        # Check if reached goal
        reward = 1.0 if next_s == goal_state else 0.0
        
        return next_s, reward
    
    # Main simulation loop
    for step in range(total_steps):
        # Check if we need to change walls at this step
        if step == change_step:
            current_walls = walls_phase2_set.copy()
        
        # 1. Select action using epsilon-greedy with random tie-breaking
        if rng.random() < epsilon:
            # Explore: choose random action
            action = rng.randint(n_actions)
        else:
            # Exploit: choose action with max Q-value, break ties randomly
            max_q = np.max(Q[state])
            # Find all actions with max Q-value
            best_actions = np.where(Q[state] == max_q)[0]
            # Randomly choose among best actions
            action = rng.choice(best_actions)
        
        # 2. Execute action in the real environment
        next_state, reward = take_action(state, action)
        
        # 3. Update Q using standard Q-learning rule
        # Q(s,a) += alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))
        if next_state == goal_state:
            best_next = 0.0
        else:
            best_next = np.max(Q[next_state])
        
        td_target = reward + gamma * best_next
        Q[state, action] += alpha * (td_target - Q[state, action])
        
        # 4. Store transition in model and record visit timestep
        model[(state, action)] = (next_state, reward)
        last_visit[(state, action)] = step
        
        # 5. Perform n_planning planning steps
        for _ in range(n_planning):
            if len(model) > 0:
                # Sample uniformly from previously seen (state, action) pairs
                idx = rng.randint(len(model))
                sampled_s, sampled_a = list(model.keys())[idx]
                sampled_next_s, sampled_reward = model[(sampled_s, sampled_a)]
                
                # Compute exploration bonus
                last_visit_step = last_visit.get((sampled_s, sampled_a), 0)
                bonus = kappa * np.sqrt(step - last_visit_step)
                
                # Apply Q-learning update with exploration bonus
                if sampled_next_s == goal_state:
                    best_next_sampled = 0.0
                else:
                    best_next_sampled = np.max(Q[sampled_next_s])
                
                td_target_sampled = sampled_reward + bonus + gamma * best_next_sampled
                Q[sampled_s, sampled_a] += alpha * (td_target_sampled - Q[sampled_s, sampled_a])
        
        # Update cumulative reward
        total_reward += reward
        cumulative_reward.append(total_reward)
        
        # Move to next state
        state = next_state
        
        # Check if reached goal
        if state == goal_state:
            episodes_completed += 1
            state = start_state  # Reset to start
    
    # Return results
    return {
        'episodes_completed': episodes_completed,
        'total_reward': round(float(total_reward), 4),
        'cumulative_reward': cumulative_reward
    }