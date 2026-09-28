import numpy as np

def target_network_update(
    online_params: list,
    target_params: list,
    tau: float = 0.005,
    update_type: str = 'soft'
) -> list:
    """
    Perform target network parameter update.
    
    Args:
        online_params: List of numpy arrays representing online network weights
        target_params: List of numpy arrays representing target network weights
        tau: Soft update interpolation coefficient (0 < tau <= 1)
        update_type: 'soft' for gradual blending, 'hard' for direct copy
    
    Returns:
        List of updated target network parameter arrays
    """
    updated_params = []
    
    if update_type == 'hard':
        # Directly copy online parameters to create independent copies
        for w_online in online_params:
            updated_params.append(w_online.copy())
            
    elif update_type == 'soft':
        # Gradually blend online parameters into the target parameters
        for w_online, w_target in zip(online_params, target_params):
            w_new = tau * w_online + (1.0 - tau) * w_target
            updated_params.append(w_new)
            
    else:
        raise ValueError(f"Unsupported update_type: '{update_type}'. Use 'soft' or 'hard'.")
        
    return updated_params