import torch

def tick_bars(ticks: torch.Tensor, bar_size: int) -> torch.Tensor:
    """
    Sample tick data into tick bars using PyTorch.
    
    Args:
        ticks: torch.Tensor of shape (n_ticks, 3) with columns [timestamp, price, volume]
        bar_size: Number of ticks per bar
    
    Returns:
        torch.Tensor of shape (n_bars, 6) with columns [timestamp, open, high, low, close, volume]
        Returns empty tensor of shape (0, 6) if input is empty or bar_size is invalid
    """
    # Validate inputs
    if ticks.numel() == 0 or bar_size <= 0:
        return torch.empty((0, 6), dtype=ticks.dtype)
    
    n_ticks = ticks.size(0)
    
    # Calculate number of complete bars
    n_bars = n_ticks // bar_size
    
    if n_bars == 0:
        return torch.empty((0, 6), dtype=ticks.dtype)
    
    # Only use complete bars worth of data
    n_used = n_bars * bar_size
    ticks = ticks[:n_used]
    
    # Reshape to group ticks into bars
    # Shape: (n_bars, bar_size, 3)
    grouped = ticks.view(n_bars, bar_size, 3)
    
    # Extract timestamp, price, volume
    timestamps = grouped[:, :, 0]  # (n_bars, bar_size)
    prices = grouped[:, :, 1]      # (n_bars, bar_size)
    volumes = grouped[:, :, 2]     # (n_bars, bar_size)
    
    # Compute bar statistics
    # timestamp: last tick in bar
    timestamp = timestamps[:, -1]  # (n_bars,)
    
    # open: first price in bar
    open_price = prices[:, 0]      # (n_bars,)
    
    # high: max price in bar
    high_price, _ = prices.max(dim=1)  # (n_bars,)
    
    # low: min price in bar
    low_price, _ = prices.min(dim=1)   # (n_bars,)
    
    # close: last price in bar
    close_price = prices[:, -1]    # (n_bars,)
    
    # volume: sum of volumes in bar
    total_volume = volumes.sum(dim=1)  # (n_bars,)
    
    # Round to 2 decimal places
    open_price = torch.round(open_price * 100) / 100
    high_price = torch.round(high_price * 100) / 100
    low_price = torch.round(low_price * 100) / 100
    close_price = torch.round(close_price * 100) / 100
    total_volume = torch.round(total_volume * 100) / 100
    
    # Stack results
    result = torch.stack([
        timestamp,
        open_price,
        high_price,
        low_price,
        close_price,
        total_volume
    ], dim=1)  # Shape: (n_bars, 6)
    
    return result