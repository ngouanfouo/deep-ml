import torch
import torch.nn as nn


def build_rope_cache(seq_len, head_dim, base=10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    positions = torch.arange(seq_len, dtype=torch.float32)
    theta = torch.outer(positions, inv_freq)  # (seq_len, head_dim // 2)
    return torch.cos(theta), torch.sin(theta)


def apply_rope(x, cos, sin):
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    rotated = torch.cat([-x2, x1], dim=-1)
    cos_full = torch.cat([cos, cos], dim=-1)
    sin_full = torch.cat([sin, sin], dim=-1)
    return x * cos_full + rotated * sin_full