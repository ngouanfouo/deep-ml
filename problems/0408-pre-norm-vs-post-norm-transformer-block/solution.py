import torch


def _layer_norm(z: torch.Tensor,
                gamma: torch.Tensor,
                beta: torch.Tensor,
                eps: float) -> torch.Tensor:
    """LayerNorm over the last axis (features), population variance."""
    mean = z.mean(dim=-1, keepdim=True)
    var = ((z - mean) ** 2).mean(dim=-1, keepdim=True)   # population variance
    z_norm = (z - mean) / torch.sqrt(var + eps)
    return z_norm * gamma + beta


def _sublayer(z: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Simplified transformer sublayer: linear projection z @ W + b."""
    return z @ W + b


def transformer_block(x: torch.Tensor,
                      W1: torch.Tensor, b1: torch.Tensor,
                      W2: torch.Tensor, b2: torch.Tensor,
                      gamma1: torch.Tensor, beta1: torch.Tensor,
                      gamma2: torch.Tensor, beta2: torch.Tensor,
                      mode: str,
                      eps: float = 1e-5) -> torch.Tensor:
    """
    Apply a transformer block with two sublayers using either pre-norm or post-norm.

    Pre-norm:  h = x + sublayer1(LN(x))
               out = h + sublayer2(LN(h))

    Post-norm: h = LN(x + sublayer1(x))
               out = LN(h + sublayer2(h))
    """
    if mode == "pre_norm":
        h = x + _sublayer(_layer_norm(x, gamma1, beta1, eps), W1, b1)
        out = h + _sublayer(_layer_norm(h, gamma2, beta2, eps), W2, b2)
    elif mode == "post_norm":
        h = _layer_norm(x + _sublayer(x, W1, b1), gamma1, beta1, eps)
        out = _layer_norm(h + _sublayer(h, W2, b2), gamma2, beta2, eps)
    else:
        raise ValueError(f"mode must be 'pre_norm' or 'post_norm', got {mode!r}")

    return out