import torch


def expert_choice_recursion(h, router_w, block, n_recursions, capacity):
    h = h.clone()  # do not modify the input tensor
    T, d = h.shape
    device = h.device

    depths = torch.zeros(T, dtype=torch.int64, device=device)
    active = torch.ones(T, dtype=torch.bool, device=device)

    for r in range(n_recursions):
        active_count = int(active.sum().item())
        if active_count == 0:
            break

        cap = capacity[r] if isinstance(capacity, (list, tuple)) else capacity
        k = min(int(cap), active_count)
        if k <= 0:
            break

        # Scores use each token's current state
        scores = torch.sigmoid(h @ router_w[r])  # (T,)
        masked_scores = scores.masked_fill(~active, float("-inf"))

        # Select top-k tokens still in play
        _, selected = torch.topk(masked_scores, k=k)

        # Update selected rows as one batch
        h_sel = h[selected]                       # (k, d)
        g_sel = scores[selected].unsqueeze(-1)    # (k, 1)
        h[selected] = h_sel + g_sel * block(h_sel)

        depths[selected] += 1

        # Only the selected tokens continue to the next recursion step
        active = torch.zeros_like(active)
        active[selected] = True

    return h, depths