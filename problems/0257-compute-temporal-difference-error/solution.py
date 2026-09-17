import torch

def compute_td_error(v_s: torch.Tensor, reward: torch.Tensor, v_s_prime: torch.Tensor, gamma: float, done: bool) -> torch.Tensor:
    """
    Compute the Temporal Difference (TD) error for a single transition.
    
    Args:
        v_s: Current state value estimate V(s) as a tensor
        reward: Immediate reward received as a tensor
        v_s_prime: Next state value estimate V(s') as a tensor
        gamma: Discount factor (0 <= gamma <= 1)
        done: True if s' is a terminal state
    
    Returns:
        The TD error delta as a tensor
    """
    v_s = torch.as_tensor(v_s, dtype=torch.float32)
    reward = torch.as_tensor(reward, dtype=torch.float32)
    v_s_prime = torch.as_tensor(v_s_prime, dtype=torch.float32)

    # TD target: no bootstrapping if the episode has terminated
    if done:
        td_target = reward
    else:
        td_target = reward + gamma * v_s_prime

    return td_target - v_s