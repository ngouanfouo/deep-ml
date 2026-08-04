import numpy as np
from multiprocessing import Pool

def simulate_episode(args: tuple) -> tuple:
    """
    Worker function for simulating a single episode in a tabular MDP.
    Must be defined at module level for pickling.
    
    Args:
        args: tuple of (worker_id, seed, max_steps, trans_probs, rewards,
              terminal_states, policy, gamma, start_state)
    
    Returns:
        Tuple of (worker_id, total_return, episode_length)
    """
    # Unpack arguments
    (worker_id, seed, max_steps, trans_probs, rewards, 
     terminal_states, policy, gamma, start_state) = args
    
    # Create deterministic random number generator for reproducibility
    rng = np.random.RandomState(seed)
    
    # Initialize episode
    state = start_state
    total_return = 0.0
    discount = 1.0
    steps = 0
    
    # Simulate episode
    for step in range(max_steps):
        # Check if current state is terminal
        if state in terminal_states:
            break
        
        # Select action using policy probabilities
        action = rng.choice(len(policy[state]), p=policy[state])
        
        # Collect reward
        reward = rewards[state, action]
        total_return += discount * reward
        discount *= gamma
        
        # Transition to next state
        next_state = rng.choice(len(trans_probs[state, action]), p=trans_probs[state, action])
        state = next_state
        steps += 1
    
    return (worker_id, total_return, steps)


def parallel_env_simulate(
    num_workers: int,
    max_steps: int,
    trans_probs: np.ndarray,
    rewards: np.ndarray,
    terminal_states: list,
    policy: np.ndarray,
    gamma: float,
    start_state: int,
    seeds: list
) -> dict:
    """
    Run parallel environment simulations using multiprocessing.
    
    Returns:
        Dictionary with keys: 'mean_return', 'std_return', 'returns', 'lengths', 'total_steps'
    """
    # Construct argument tuples for each worker
    args_list = []
    for worker_id in range(num_workers):
        args = (
            worker_id,
            seeds[worker_id],
            max_steps,
            trans_probs,
            rewards,
            terminal_states,
            policy,
            gamma,
            start_state
        )
        args_list.append(args)
    
    # Run simulations in parallel
    with Pool(processes=num_workers) as pool:
        results = pool.map(simulate_episode, args_list)
    
    # Sort results by worker_id to ensure deterministic ordering
    results.sort(key=lambda x: x[0])
    
    # Extract returns and lengths
    returns = [float(r[1]) for r in results]  # Convert to Python float
    lengths = [int(r[2]) for r in results]
    
    # Compute statistics
    total_steps = sum(lengths)
    mean_return = np.mean(returns)
    std_return = np.std(returns)  # population standard deviation
    
    # Round to 4 decimal places
    mean_return = np.round(mean_return, 4)
    std_return = np.round(std_return, 4)
    returns_rounded = [np.round(r, 4) for r in returns]
    
    return {
        'mean_return': float(mean_return),
        'std_return': float(std_return),
        'returns': [float(r) for r in returns_rounded],  # Ensure Python floats
        'lengths': lengths,
        'total_steps': total_steps
    }