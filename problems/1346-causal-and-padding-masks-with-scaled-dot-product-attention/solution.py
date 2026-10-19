import torch
import torch.nn as nn
import torch.nn.functional as F


def attention(q, k, v, key_padding_mask=None, causal=False):
    T_q = q.shape[-2]
    T_k = k.shape[-2]

    attn_mask = None

    if causal:
        causal_mask = torch.ones(T_q, T_k, dtype=torch.bool, device=q.device).tril()
        attn_mask = causal_mask

    if key_padding_mask is not None:
        # (B, T_k) -> (B, 1, 1, T_k) so it broadcasts over heads and queries
        pad = key_padding_mask.view(key_padding_mask.shape[0], 1, 1, T_k)
        attn_mask = pad if attn_mask is None else (attn_mask & pad)

    return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)