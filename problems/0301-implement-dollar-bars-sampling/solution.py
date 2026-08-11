import torch

def dollar_bars(trades: torch.Tensor, dollar_threshold: float) -> torch.Tensor:
    """
    Generate dollar bars from trade data.
    
    Args:
        trades: torch.Tensor of shape (N, 2) where each row is (price, volume)
        dollar_threshold: Dollar amount threshold for creating a new bar
        
    Returns:
        torch.Tensor of shape (M, 6) representing dollar bars
        (open, high, low, close, volume, dollar_value)
    """
    # Input validation
    if trades.numel() == 0 or dollar_threshold <= 0:
        return torch.empty((0, 6), dtype=torch.float64)
    
    # Extract prices and volumes
    prices = trades[:, 0]  # (N,)
    volumes = trades[:, 1]  # (N,)
    
    # Calculate dollar values for each trade
    dollar_values = prices * volumes  # (N,)
    
    # Initialize variables for bar construction
    bars = []
    current_open = None
    current_high = None
    current_low = None
    current_close = None
    current_volume = 0.0
    current_dollar = 0.0
    bar_started = False
    
    # Iterate through trades
    for i in range(len(trades)):
        price = prices[i].item()
        volume = volumes[i].item()
        dollar_value = dollar_values[i].item()
        
        # Start a new bar if needed
        if not bar_started:
            current_open = price
            current_high = price
            current_low = price
            current_close = price
            current_volume = volume
            current_dollar = dollar_value
            bar_started = True
        else:
            # Update current bar
            current_high = max(current_high, price)
            current_low = min(current_low, price)
            current_close = price
            current_volume += volume
            current_dollar += dollar_value
        
        # Check if dollar threshold is reached or exceeded
        if current_dollar >= dollar_threshold:
            # Round to 4 decimal places
            bar = torch.tensor([
                round(current_open, 4),
                round(current_high, 4),
                round(current_low, 4),
                round(current_close, 4),
                round(current_volume, 4),
                round(current_dollar, 4)
            ], dtype=torch.float64)
            bars.append(bar)
            
            # Reset for next bar
            bar_started = False
            current_volume = 0.0
            current_dollar = 0.0
    
    # Return results as a tensor
    if bars:
        return torch.stack(bars)
    else:
        return torch.empty((0, 6), dtype=torch.float64)