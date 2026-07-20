import numpy as np

def dyna_q(num_states, num_actions, P, R, terminal_states, alpha, gamma, epsilon, num_episodes, n_planning):
    """
    Implement the Dyna-Q algorithm.
    
    Args:
        num_states: int, number of states
        num_actions: int, number of actions
        P: np.ndarray of shape (num_states, num_actions, num_states), transition probabilities
        R: np.ndarray of shape (num_states, num_actions), rewards
        terminal_states: list of int, terminal state indices
        alpha: float, learning rate
        gamma: float, discount factor
        epsilon: float, exploration rate
        num_episodes: int, number of episodes
        n_planning: int, number of planning steps per real step
    
    Returns:
        Q: np.ndarray of shape (num_states, num_actions), learned Q-table
    """
    # Initialize the Q-table to zeros
    Q = np.zeros((num_states, num_actions), dtype=np.float64)
    
    # Track observed state-action pairs for uniform sampling during planning
    # A list of tuples: (state, action)
    observed_sa_pairs = []
    # To check uniqueness efficiently
    observed_sa_set = set()
    
    # Model storage: maps (state, action) -> (reward, next_state)
    model = {}
    
    terminal_set = set(terminal_states)
    non_terminal_states = [s for s in range(num_states) if s not in terminal_set]
    
    # If no valid starting state exists, return early
    if not non_terminal_states:
        return Q

    for _ in range(num_episodes):
        # 1. Start each episode from a randomly chosen non-terminal state
        state = np.random.choice(non_terminal_states)
        
        while state not in terminal_set:
            # 2. Epsilon-greedy action selection
            if np.random.uniform(0, 1) < epsilon:
                action = np.random.randint(0, num_actions)
            else:
                # Greedy selection: break ties by picking the lowest index
                state_qs = Q[state]
                max_q = np.max(state_qs)
                action = int(np.min(np.where(state_qs == max_q)[0]))
            
            # 3. Environment Step (Sampling next state based on true P and R matrices)
            probs = P[state, action]
            next_state = np.random.choice(num_states, p=probs)
            reward = R[state, action]
            
            # 4. Direct RL update (Q-learning rule)
            if next_state in terminal_set:
                max_next_q = 0.0
            else:
                max_next_q = np.max(Q[next_state])
                
            Q[state, action] += alpha * (reward + gamma * max_next_q - Q[state, action])
            
            # 5. Model Learning: store the most recently observed transition
            model[(state, action)] = (reward, next_state)
            if (state, action) not in observed_sa_set:
                observed_sa_set.add((state, action))
                observed_sa_pairs.append((state, action))
                
            # 6. Planning Phase
            if n_planning > 0 and len(observed_sa_pairs) > 0:
                for _ in range(n_planning):
                    # Randomly sample an already experienced state-action pair
                    idx = np.random.randint(0, len(observed_sa_pairs))
                    sim_state, sim_action = observed_sa_pairs[idx]
                    
                    # Get predicted values from the deterministic model
                    sim_reward, sim_next_state = model[(sim_state, sim_action)]
                    
                    # Update Q-table using the simulated transition
                    if sim_next_state in terminal_set:
                        sim_max_next_q = 0.0
                    else:
                        sim_max_next_q = np.max(Q[sim_next_state])
                        
                    Q[sim_state, sim_action] += alpha * (sim_reward + gamma * sim_max_next_q - Q[sim_state, sim_action])
            
            # Advance state variable
            state = next_state
            
    return Q