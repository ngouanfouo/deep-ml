import numpy as np

def gambler_value_iteration(ph, theta=1e-9):
    """
    Computes the optimal value function and policy for the Gambler's Problem.
    Args:
      ph: probability of heads
      theta: convergence threshold
    Returns:
      V: list of values for all states 0..100
      policy: list of optimal stakes for all states 0..100
    """
    # Initialize V with zeros for all states 0..100
    V = np.zeros(101)
    policy = np.zeros(101, dtype=int)
    
    # Terminal states: V[0] = 0, V[100] = 0 (no value since game ends)
    # We'll keep them at 0
    
    # Value iteration
    while True:
        delta = 0
        # Iterate over all non-terminal states
        for s in range(1, 100):
            # Store old value for convergence check
            old_v = V[s]
            
            # Calculate maximum value over all possible actions
            # Action a = stake, from 1 to min(s, 100-s)
            max_value = -float('inf')
            best_action = 1
            
            max_stake = min(s, 100 - s)
            for a in range(1, max_stake + 1):
                # Calculate expected value for this action
                # If heads: transition to s + a
                # If tails: transition to s - a
                # Reward is +1 if we reach 100 in a transition
                
                # Head transition
                if s + a == 100:
                    head_value = 1.0  # Reward for reaching goal
                else:
                    head_value = V[s + a]
                
                # Tail transition
                if s - a == 0:
                    tail_value = 0.0  # Bankruptcy, no reward
                else:
                    tail_value = V[s - a]
                
                # Expected value
                expected_value = ph * head_value + (1 - ph) * tail_value
                
                # Keep track of best action
                if expected_value > max_value:
                    max_value = expected_value
                    best_action = a
            
            # Update V[s] with the maximum value
            V[s] = max_value
            policy[s] = best_action
            
            # Update delta for convergence check
            delta = max(delta, abs(old_v - V[s]))
        
        # Check for convergence
        if delta < theta:
            break
    
    # Convert to lists for return
    V = V.tolist()
    policy = policy.tolist()
    
    return V, policy