import torch

def sequence_mask(values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Zero out positions at or beyond each row's length."""
    batch, seq = values.shape
    # Positions along the sequence dimension, shape (seq,)
    positions = torch.arange(seq, device=values.device)
    # Broadcast: lengths (batch, 1) > positions (1, seq) -> mask (batch, seq)
    mask = positions.unsqueeze(0) < lengths.unsqueeze(1)
    return values * mask