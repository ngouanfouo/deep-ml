import numpy as np

def compute_eligibility_traces(
    state_visits: list,
    num_states: int,
    gamma: float,
    lam: float,
    trace_type: str,
    alpha: float = 0.1
) -> list:
    """
    Compute the final eligibility trace vector after processing a sequence of state visits.
    
    Args:
        state_visits: List of state indices visited at each timestep
        num_states: Total number of states in the environment
        gamma: Discount factor
        lam: Trace decay parameter (lambda)
        trace_type: One of 'accumulating', 'replacing', or 'dutch'
        alpha: Step-size parameter (used only for Dutch traces)
    
    Returns:
        Final eligibility trace vector as a list of floats
    """
    e = np.zeros(num_states, dtype=float)
    decay = gamma * lam

    for s in state_visits:
        # 1. Decay all traces by gamma * lambda
        e *= decay

        # 2. Update the trace for the visited state
        if trace_type == 'accumulating':
            e[s] += 1.0
        elif trace_type == 'replacing':
            e[s] = 1.0
        elif trace_type == 'dutch':
            e[s] += alpha * (1.0 - e[s])
        else:
            raise ValueError(
                f"Unknown trace_type '{trace_type}'. "
                "Expected 'accumulating', 'replacing', or 'dutch'."
            )

    return e.tolist()