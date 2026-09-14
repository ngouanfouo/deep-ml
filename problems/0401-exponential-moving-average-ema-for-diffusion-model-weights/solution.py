import torch

def ema_update(ema_params: torch.Tensor, model_params_list: list, decay: float) -> torch.Tensor:
    """
    Compute the Exponential Moving Average of model parameters over training steps.

    Args:
        ema_params: torch.Tensor, initial EMA parameters
        model_params_list: list of torch.Tensors, model params at each training step
        decay: float, EMA decay rate in [0, 1]
    Returns:
        Final EMA parameters as a torch.Tensor, rounded to 4 decimal places
    """
    # Create a clone of the initial EMA parameters to avoid in-place modification of inputs
    ema = ema_params.clone().detach()
    
    # Iterate through each parameter snapshot in order
    for model_params in model_params_list:
        # Apply the EMA update rule: ema = decay * ema + (1 - decay) * model_params
        ema = decay * ema + (1.0 - decay) * model_params
        
    # Round final parameters to 4 decimal places
    return torch.round(ema, decimals=4)