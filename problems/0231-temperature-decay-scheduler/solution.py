import numpy as np

def temperature_decay(
    schedule_type: str,
    initial_temp: float,
    current_step: int,
    total_steps: int,
    final_temp: float = 0.01,
    decay_rate: float = 0.95
) -> float:
    """
    Compute temperature at current training step using decay schedule.
    
    Temperature controls randomness in neural network outputs:
    - High temperature: More random, more exploration
    - Low temperature: More deterministic, more exploitation
    
    Args:
        schedule_type: Decay schedule type
          'linear': Steady linear decrease
          'exponential': Fast early decay, slow later
          'cosine': Smooth cosine curve
          'constant': No decay
        initial_temp: Starting temperature
        current_step: Current training step (0 to total_steps)
        total_steps: Total number of training steps
        final_temp: Minimum temperature (floor)
        decay_rate: Decay rate per step (for exponential)
    
    Returns:
        Temperature value at current step
    """
    # Clamp current_step to valid range
    current_step = max(0, min(current_step, total_steps))
    
    if schedule_type == 'constant':
        return float(initial_temp)
    
    elif schedule_type == 'linear':
        # Linear interpolation from initial_temp to final_temp
        progress = current_step / total_steps if total_steps > 0 else 1.0
        temperature = initial_temp - (initial_temp - final_temp) * progress
        return float(max(temperature, final_temp))
    
    elif schedule_type == 'exponential':
        # Exponential decay: T = initial_temp * (decay_rate ^ step)
        temperature = initial_temp * (decay_rate ** current_step)
        return float(max(temperature, final_temp))
    
    elif schedule_type == 'cosine':
        # Cosine annealing: smooth decrease following cosine curve
        # T = final_temp + 0.5 * (initial_temp - final_temp) * (1 + cos(π * step/total_steps))
        if total_steps == 0:
            return float(initial_temp)
        progress = current_step / total_steps
        cosine_val = 0.5 * (1 + np.cos(np.pi * progress))
        temperature = final_temp + (initial_temp - final_temp) * cosine_val
        return float(temperature)
    
    else:
        raise ValueError(f"Unknown schedule_type: {schedule_type}. "
                         f"Must be one of: 'linear', 'exponential', 'cosine', 'constant'")