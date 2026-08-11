import numpy as np

def volume_bars_sampling(prices: np.ndarray, volumes: np.ndarray, volume_threshold: float) -> list:
    """
    Generate volume bars from tick/trade data.
    
    Args:
        prices: Array of trade prices for each trade/tick
        volumes: Array of trade volumes corresponding to each price
        volume_threshold: Volume threshold that triggers formation of a new bar
    
    Returns:
        List of bars, where each bar is [open, high, low, close, total_volume]
        All values rounded to 4 decimal places
    """
    # Input validation
    if len(prices) == 0 or len(volumes) == 0 or len(prices) != len(volumes) or volume_threshold <= 0:
        return []
    
    bars = []
    
    # Initialize current bar
    current_open = None
    current_high = None
    current_low = None
    current_close = None
    current_volume = 0.0
    bar_started = False
    
    # Iterate through each trade
    for price, volume in zip(prices, volumes):
        # Start a new bar if needed
        if not bar_started:
            current_open = price
            current_high = price
            current_low = price
            current_close = price
            current_volume = volume
            bar_started = True
        else:
            # Update current bar with new trade
            current_high = max(current_high, price)
            current_low = min(current_low, price)
            current_close = price
            current_volume += volume
        
        # Check if volume threshold is reached
        if current_volume >= volume_threshold:
            # Round values to 4 decimal places
            bar = [
                round(current_open, 4),
                round(current_high, 4),
                round(current_low, 4),
                round(current_close, 4),
                round(current_volume, 4)
            ]
            bars.append(bar)
            
            # Reset for next bar
            bar_started = False
            current_volume = 0.0
    
    # Add incomplete bar if there's remaining data
    if bar_started and current_volume > 0:
        bar = [
            round(current_open, 4),
            round(current_high, 4),
            round(current_low, 4),
            round(current_close, 4),
            round(current_volume, 4)
        ]
        bars.append(bar)
    
    return bars