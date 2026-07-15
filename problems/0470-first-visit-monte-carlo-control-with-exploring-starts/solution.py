import numpy as np

def mc_control_exploring_starts(env: dict, n_states: int, n_actions: int, gamma: float, n_episodes: int, max_steps: int = 100, seed: int = 42) -> tuple:
    """
    First-visit Monte Carlo control with exploring starts.
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Initialize Q-values to zeros
    Q = np.zeros((n_states, n_actions))
    
    # Initialize returns sums and counts for incremental mean updates
    returns_sum = np.zeros((n_states, n_actions))
    returns_count = np.zeros((n_states, n_actions))
    
    # Initialize policy randomly
    policy = np.random.randint(0, n_actions, size=n_states)
    
    for episode in range(n_episodes):
        # Exploring starts: choose random initial state and action
        state = np.random.randint(0, n_states)
        action = np.random.randint(0, n_actions)
        
        # Generate an episode by following current policy
        episode_states = []
        episode_actions = []
        episode_rewards = []
        
        # Start with the exploring start
        current_state = state
        current_action = action
        
        for step in range(max_steps):
            # Store the state-action pair
            episode_states.append(current_state)
            episode_actions.append(current_action)
            
            # Take the action and observe next state and reward
            if (current_state, current_action) not in env:
                # Invalid action: end episode
                episode_rewards.append(0.0)  # No transition, return 0 reward
                break
            
            next_state, reward, done = env[(current_state, current_action)]
            episode_rewards.append(reward)
            
            if done:
                break
            
            # Select next action using current policy
            current_state = next_state
            current_action = policy[current_state]
        
        # First-visit MC update: process episode backwards
        G = 0.0
        visited_pairs = set()
        
        for t in range(len(episode_rewards) - 1, -1, -1):
            G = gamma * G + episode_rewards[t]
            state_t = episode_states[t]
            action_t = episode_actions[t]
            
            # Only update if this is the first visit to this state-action pair
            if (state_t, action_t) not in visited_pairs:
                visited_pairs.add((state_t, action_t))
                
                # Incremental mean update
                returns_sum[state_t, action_t] += G
                returns_count[state_t, action_t] += 1
                Q[state_t, action_t] = returns_sum[state_t, action_t] / returns_count[state_t, action_t]
        
        # Policy improvement: update policy greedily
        for s in range(n_states):
            policy[s] = np.argmax(Q[s])
    
    return Q, policy