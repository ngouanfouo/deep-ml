import torch


def per_channel_quantize(weight: torch.Tensor, bits: int = 8) -> tuple:
    """
    Perform symmetric per-channel post-training quantization.

    Args:
        weight: Weight matrix tensor of shape (out_channels, in_features)
        bits: Target bit-width for quantization (default: 8)

    Returns:
        Tuple of (quantized_weights, scale_factors, dequantized_weights)
    """
    # Symmetric integer range: [-(2^(bits-1)), 2^(bits-1) - 1]
    qmax = 2 ** (bits - 1) - 1
    qmin = -(2 ** (bits - 1))

    # Per-channel (per-row) max absolute value
    max_abs = weight.abs().amax(dim=1)                 # (out_channels,)

    # Scale maps max|w| to qmax. Guard all-zero channels with scale 1.0.
    scale = torch.where(
        max_abs > 0,
        max_abs / float(qmax),
        torch.ones_like(max_abs),
    )                                                  # (out_channels,)

    # Quantize: divide, round, clip to valid integer range
    scale_exp = scale.unsqueeze(1)                     # (out_channels, 1)
    quantized = torch.round(weight / scale_exp)
    quantized = torch.clamp(quantized, qmin, qmax).to(torch.int32)

    # Dequantize: back to float using the same per-channel scale
    dequantized = quantized.to(weight.dtype) * scale_exp

    return quantized, scale, dequantized