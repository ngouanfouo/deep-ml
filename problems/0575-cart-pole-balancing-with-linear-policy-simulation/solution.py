import numpy as np

def simulate_cartpole(initial_state: list, policy_weights: list, max_steps: int = 200) -> tuple:
    """
    Simulate cart-pole balancing with a linear policy.
    
    Args:
        initial_state: List of 4 floats [x, x_dot, theta, theta_dot]
        policy_weights: List of 4 floats for the linear policy
        max_steps: Maximum number of simulation steps
    
    Returns:
        Tuple of (total_reward, final_state) where total_reward is an int
        and final_state is a list of 4 floats rounded to 4 decimal places.
    """
    # Physical constants
    gravity = 9.8
    cart_mass = 1.0
    pole_mass = 0.1
    pole_half_length = 0.5
    dt = 0.02
    force_magnitude = 10.0
    
    # Bounds for termination
    x_bound = 2.4
    theta_bound = 12 * np.pi / 180.0  # 12 degrees in radians
    
    # Convert to numpy arrays for efficient computation
    state = np.array(initial_state, dtype=np.float64)
    weights = np.array(policy_weights, dtype=np.float64)
    
    total_reward = 0
    done = False
    
    for step in range(max_steps):
        # Check termination at start of step (before applying action)
        x, x_dot, theta, theta_dot = state
        
        if abs(x) > x_bound or abs(theta) > theta_bound:
            done = True
            break
        
        # Select action using linear policy
        dot_product = np.dot(weights, state)
        if dot_product >= 0:
            action = 1  # Push right (+10.0 N)
            force = force_magnitude
        else:
            action = 0  # Push left (-10.0 N)
            force = -force_magnitude
        
        # Dynamics computation
        # Mass terms
        total_mass = cart_mass + pole_mass
        pole_mass_half_length = pole_mass * pole_half_length
        
        # Compute accelerations using the cart-pole equations
        # From standard cart-pole dynamics:
        # theta_ddot = (g * sin(theta) - cos(theta) * (force + pole_mass * pole_half_length * theta_dot^2 * sin(theta)) / total_mass)
        #             / (pole_half_length * (4/3 - pole_mass * cos(theta)^2 / total_mass))
        # x_ddot = (force + pole_mass * pole_half_length * (theta_dot^2 * sin(theta) - theta_ddot * cos(theta))) / total_mass
        
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        
        # Calculate theta acceleration
        numerator = gravity * sin_theta - cos_theta * (force + pole_mass * pole_half_length * theta_dot**2 * sin_theta) / total_mass
        denominator = pole_half_length * (4.0/3.0 - pole_mass * cos_theta**2 / total_mass)
        theta_ddot = numerator / denominator
        
        # Calculate x acceleration
        x_ddot = (force + pole_mass * pole_half_length * (theta_dot**2 * sin_theta - theta_ddot * cos_theta)) / total_mass
        
        # Euler integration (single step)
        state[0] += state[1] * dt  # x += x_dot * dt
        state[1] += x_ddot * dt    # x_dot += x_ddot * dt
        state[2] += state[3] * dt  # theta += theta_dot * dt
        state[3] += theta_ddot * dt  # theta_dot += theta_ddot * dt
        
        # Reward for surviving this step
        total_reward += 1
    
    # Final state rounded to 4 decimal places
    final_state = [round(float(val), 4) for val in state]
    
    return total_reward, final_state