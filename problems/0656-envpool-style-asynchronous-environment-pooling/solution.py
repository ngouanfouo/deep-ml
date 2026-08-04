import numpy as np

def async_env_pool(num_envs: int, step_durations: list, rewards_per_step: list, total_ticks: int, batch_size: int) -> tuple:
    """
    Simulate an asynchronous environment pool.

    Args:
        num_envs: Number of environments in the pool
        step_durations: List of step durations (in ticks) for each env
        rewards_per_step: List of rewards for each env per completed step
        total_ticks: Total simulation time in ticks
        batch_size: Max environments to dispatch per tick

    Returns:
        Tuple of (total_steps, total_reward, avg_envs_active, throughput)
    """
    # Handle edge case
    if total_ticks == 0:
        return (0, 0.0, 0.0, 0.0)
    
    # Initialize environment states
    # remaining_time[i] = 0 means environment i is ready; > 0 means busy with countdown
    remaining_time = [0] * num_envs
    
    # Track cumulative statistics
    total_steps = 0
    total_reward = 0.0
    busy_counts = []
    
    # Simulate each tick
    for tick in range(total_ticks):
        # Phase 1: Advance all busy environments by one tick
        # Any environment that reaches 0 completes its step
        for i in range(num_envs):
            if remaining_time[i] > 0:
                remaining_time[i] -= 1
                # If completed (reached 0), collect reward
                if remaining_time[i] == 0:
                    total_steps += 1
                    total_reward += rewards_per_step[i]
        
        # Phase 2: Dispatch new actions to ready environments
        # Select ready environments in ascending index order
        ready_indices = [i for i in range(num_envs) if remaining_time[i] == 0]
        
        # Dispatch up to batch_size ready environments
        num_to_dispatch = min(batch_size, len(ready_indices))
        for i in range(num_to_dispatch):
            env_idx = ready_indices[i]
            # Set remaining time to full duration
            remaining_time[env_idx] = step_durations[env_idx]
        
        # Phase 3: Record the number of currently busy environments
        busy_count = sum(1 for t in remaining_time if t > 0)
        busy_counts.append(busy_count)
    
    # Compute statistics
    avg_envs_active = np.mean(busy_counts) if busy_counts else 0.0
    throughput = total_steps / total_ticks
    
    # Round to 4 decimal places
    total_reward = np.round(total_reward, 4)
    avg_envs_active = np.round(avg_envs_active, 4)
    throughput = np.round(throughput, 4)
    
    return (total_steps, float(total_reward), float(avg_envs_active), float(throughput))