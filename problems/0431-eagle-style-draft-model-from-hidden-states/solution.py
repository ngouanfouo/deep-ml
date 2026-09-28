import torch
import torch.nn.functional as F


def eagle_draft_forward(
    hidden_state: torch.Tensor,
    token_id: int,
    embed_matrix: torch.Tensor,
    fc_fuse_weight: torch.Tensor,
    fc_fuse_bias: torch.Tensor,
    draft_head_weight: torch.Tensor,
    draft_head_bias: torch.Tensor,
    lm_head_weight: torch.Tensor,
    num_draft_tokens: int = 3
) -> list:
    """
    Generate draft tokens using an EAGLE-style draft model.

    Autoregressively produces `num_draft_tokens` token IDs. At each step:
        embed    = embed_matrix[token_id]                # (d,)
        concat   = [hidden_state ; embed]                # (2d,)
        fused    = ReLU(concat @ fc_fuse_weight.T + fc_fuse_bias)   # (d,)
        h        = ReLU(fused  @ draft_head_weight.T + draft_head_bias)  # (d,)
        logits   = lm_head_weight @ h                    # (V,)
        token_id = argmax(logits)
        hidden_state = h
    """
    draft_tokens = []
    h = hidden_state

    for _ in range(num_draft_tokens):
        # 1) Token embedding for the current token
        embed = embed_matrix[token_id]                          # (d,)

        # 2) Concatenate hidden state with token embedding
        concat = torch.cat([h, embed], dim=-1)                  # (2d,)

        # 3) Fusion layer: linear + ReLU
        fused = torch.relu(concat @ fc_fuse_weight.t() + fc_fuse_bias)   # (d,)

        # 4) Draft head: linear + ReLU
        h = torch.relu(fused @ draft_head_weight.t() + draft_head_bias)  # (d,)

        # 5) Project to vocabulary logits
        logits = lm_head_weight @ h                             # (V,)

        # 6) Greedy decode (argmax picks the first index on ties)
        token_id = int(torch.argmax(logits).item())
        draft_tokens.append(token_id)

    return draft_tokens