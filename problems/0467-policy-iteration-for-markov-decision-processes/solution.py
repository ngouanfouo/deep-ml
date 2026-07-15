import numpy as np

def policy_iteration(num_states: int, num_actions: int, transitions: list, gamma: float, theta: float = 1e-8) -> tuple:
    """
    Implement the Policy Iteration algorithm for solving MDPs.
    """
    # Initialize policy: action 0 for every state
    policy = [0] * num_states
    
    # Initialize value function: all zeros
    values = [0.0] * num_states
    
    while True:
        # Policy Evaluation: Iteratively compute values for current policy
        while True:
            delta = 0.0
            new_values = values.copy()
            
            for s in range(num_states):
                a = policy[s]
                # Compute expected value for taking action a in state s
                expected_value = 0.0
                for prob, next_state, reward in transitions[s][a]:
                    expected_value += prob * (reward + gamma * values[next_state])
                new_values[s] = expected_value
                
                delta = max(delta, abs(new_values[s] - values[s]))
            
            values = new_values
            
            # Check convergence
            if delta < theta:
                break
        
        # Policy Improvement: Update policy greedily
        policy_stable = True
        
        for s in range(num_states):
            # Compute Q-values for all actions
            q_values = []
            for a in range(num_actions):
                q_value = 0.0
                for prob, next_state, reward in transitions[s][a]:
                    q_value += prob * (reward + gamma * values[next_state])
                q_values.append(q_value)
            
            # Choose action with maximum Q-value (smallest index for ties)
            best_action = np.argmax(q_values)
            
            # Check if policy changed
            if best_action != policy[s]:
                policy_stable = False
                policy[s] = best_action
        
        # If policy is stable, we're done
        if policy_stable:
            break
    
    # Round values to 4 decimal places
    values = [round(v, 4) for v in values]
    
    return policy, values