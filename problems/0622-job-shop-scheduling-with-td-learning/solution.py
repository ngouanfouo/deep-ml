import numpy as np
from collections import defaultdict

def jobshop_td_learning(jobs, n_machines, n_episodes, gamma, alpha, epsilon, seed=42):
    """
    Learn a scheduling policy for job-shop scheduling using SARSA (TD(0)).

    Args:
        jobs: list of lists, jobs[i] = [(machine, duration), ...] for job i
        n_machines: int, number of machines
        n_episodes: int, number of training episodes
        gamma: float, discount factor
        alpha: float, learning rate
        epsilon: float, exploration rate
        seed: int, random seed

    Returns:
        dict with 'best_makespan', 'best_schedule', 'q_values'
    """
    rng = np.random.RandomState(seed)
    n_jobs = len(jobs)
    
    # Get total operations for each job
    job_lengths = [len(job) for job in jobs]
    
    # Initialize Q-table
    Q = defaultdict(float)
    
    best_makespan = float('inf')
    best_schedule = None
    
    def get_actions(state):
        """Return list of available actions (jobs that still have operations left)."""
        return [j for j in range(n_jobs) if state[j] < job_lengths[j]]
    
    def get_next_operation(job, state):
        """Get the next operation for a job given its current progress."""
        op_idx = state[job]
        return jobs[job][op_idx]
    
    def schedule_operation(job, machine_times, job_times, state):
        """Schedule the next operation for a job and return updated times."""
        op_idx = state[job]
        machine, duration = jobs[job][op_idx]
        
        # Get earliest start time: max(machine availability, job availability)
        start_time = max(machine_times[machine], job_times[job])
        
        # Schedule the operation
        end_time = start_time + duration
        machine_times[machine] = end_time
        job_times[job] = end_time
        
        return start_time, end_time
    
    def compute_makespan(machine_times, job_times):
        """Compute makespan as the maximum completion time across all machines and jobs."""
        return max(max(machine_times), max(job_times))
    
    def run_episode():
        """Run one episode of SARSA learning."""
        # Initialize state
        state = tuple([0] * n_jobs)
        schedule = []
        machine_times = np.zeros(n_machines)
        job_times = np.zeros(n_jobs)
        
        # Choose first action using epsilon-greedy
        actions = get_actions(state)
        if rng.random() < epsilon:
            action = rng.choice(actions)
        else:
            # Greedy with tie-breaking by lowest index
            q_values = [Q[(state, a)] for a in actions]
            max_q = max(q_values)
            # Get all actions with max Q-value
            best_actions = [a for a, q in zip(actions, q_values) if q == max_q]
            action = min(best_actions)  # Tie-break by lowest index
        
        while True:
            # Store current state and action
            current_state = state
            current_action = action
            
            # Schedule the operation
            job = int(current_action)  # Ensure we use Python int
            schedule.append(job)
            schedule_operation(job, machine_times, job_times, list(state))
            state = list(state)
            state[job] += 1
            state = tuple(state)
            
            # Check if terminal
            is_terminal = all(state[j] == job_lengths[j] for j in range(n_jobs))
            
            if is_terminal:
                # Terminal transition: reward = -makespan
                makespan = compute_makespan(machine_times, job_times)
                reward = -makespan
                next_action = None
                
                # Update Q-value for terminal transition
                Q[(current_state, current_action)] += alpha * (reward - Q[(current_state, current_action)])
                
                return makespan, schedule
            else:
                # Non-terminal transition: reward = 0
                reward = 0
                
                # Choose next action using epsilon-greedy
                next_actions = get_actions(state)
                if rng.random() < epsilon:
                    next_action = rng.choice(next_actions)
                else:
                    q_values = [Q[(state, a)] for a in next_actions]
                    max_q = max(q_values)
                    best_actions = [a for a, q in zip(next_actions, q_values) if q == max_q]
                    next_action = min(best_actions)
                
                # SARSA update
                Q[(current_state, current_action)] += alpha * (reward + gamma * Q[(state, next_action)] - Q[(current_state, current_action)])
                
                # Move to next state-action pair
                state = state
                action = next_action
    
    # Training loop
    for episode in range(n_episodes):
        makespan, schedule = run_episode()
        
        if makespan < best_makespan:
            best_makespan = makespan
            best_schedule = schedule
    
    # Round Q-values to 4 decimal places
    q_values_rounded = {k: round(v, 4) for k, v in Q.items()}
    
    # Ensure best_schedule contains Python ints
    if best_schedule is not None:
        best_schedule = [int(job) for job in best_schedule]
    
    return {
        'best_makespan': round(best_makespan, 4),
        'best_schedule': best_schedule,
        'q_values': q_values_rounded
    }