import numpy as np
import heapq

def prioritized_sweeping(num_states: int, num_actions: int, experiences: list, alpha: float, gamma: float, theta: float, n_planning: int) -> np.ndarray:
    """
    Implement the Prioritized Sweeping algorithm for model-based RL.
    
    Args:
        num_states: Number of states
        num_actions: Number of actions
        experiences: List of (state, action, reward, next_state, done) tuples
        alpha: Learning rate
        gamma: Discount factor
        theta: Minimum priority threshold
        n_planning: Maximum number of planning steps per real experience
        
    Returns:
        Q: np.ndarray of shape (num_states, num_actions), final action-value table
    """
    # Initialize the action-value table to zeros
    Q = np.zeros((num_states, num_actions), dtype=np.float64)
    
    # Model storage: maps (state, action) -> (reward, next_state, done)
    model = {}
    
    # Predecessor tracking: maps a state s to a set of (s_bar, a_bar) pairs that lead to s
    predecessors = {s: set() for s in range(num_states)}
    
    for state, action, reward, next_state, done in experiences:
        # 1. Update the environment model with the latest real experience
        old_transition = model.get((state, action))
        if old_transition is not None:
            # If the transition changed, clean up the old predecessor mapping
            old_next_state = old_transition[1]
            predecessors[old_next_state].discard((state, action))
            
        model[(state, action)] = (reward, next_state, done)
        predecessors[next_state].add((state, action))
        
        # 2. Compute the absolute TD error for the real transition
        if done:
            max_next_q = 0.0
        else:
            max_next_q = np.max(Q[next_state])
            
        target = reward + gamma * max_next_q
        priority = abs(target - Q[state, action])
        
        # Initialize the priority queue structures for this planning loop
        pq = []
        # Track the highest priority currently assigned to each pair in the queue
        pq_entries = {}
        
        if priority > theta:
            heapq.heappush(pq, (-priority, state, action))
            pq_entries[(state, action)] = priority
            
        # 3. Planning phase
        planning_steps = 0
        while pq and planning_steps < n_planning:
            neg_p, s, a = heapq.heappop(pq)
            p = -neg_p
            
            # Skip this element if a higher priority update for (s, a) was processed already
            if pq_entries.get((s, a), 0.0) != p:
                continue
                
            # Remove the pair from the dictionary since it is now being processed
            del pq_entries[(s, a)]
            
            # Perform the Q-value update for the popped state-action pair
            r_sim, s_next_sim, done_sim = model[(s, a)]
            if done_sim:
                max_sim_q = 0.0
            else:
                max_sim_q = np.max(Q[s_next_sim])
                
            Q[s, a] += alpha * (r_sim + gamma * max_sim_q - Q[s, a])
            planning_steps += 1
            
            # 4. Propagate priority to all predecessors of the updated state `s`
            for s_bar, a_bar in predecessors[s]:
                r_pred, _, done_pred = model[(s_bar, a_bar)]
                
                # Check target value of the predecessor transition leading into `s`
                # Note: done_pred should generally be False if it transitions into s, but handled for safety
                max_q_s = np.max(Q[s]) if not done_pred else 0.0
                
                target_pred = r_pred + gamma * max_q_s
                p_pred = abs(target_pred - Q[s_bar, a_bar])
                
                if p_pred > theta:
                    # If it's already in the queue, only update if the new priority is larger
                    if p_pred > pq_entries.get((s_bar, a_bar), 0.0):
                        pq_entries[(s_bar, a_bar)] = p_pred
                        heapq.heappush(pq, (-p_pred, s_bar, a_bar))
                        
    return Q