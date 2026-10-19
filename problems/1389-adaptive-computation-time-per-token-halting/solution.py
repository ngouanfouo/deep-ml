import torch


def act_halting(halt_probs, threshold=0.99):
    T, S = halt_probs.shape

    cum = torch.cumsum(halt_probs, dim=1)          # (T, S)
    reached = cum >= threshold                     # (T, S) bool
    has_reached = reached.any(dim=1)               # (T,)

    first_idx = reached.int().argmax(dim=1)        # first True, or 0 if none
    last_idx = torch.full_like(first_idx, S - 1)
    N = torch.where(has_reached, first_idx, last_idx)  # halting loop per token

    n_steps = N + 1                                # loops actually run

    # Sum of probabilities strictly before the halting loop (0 if N == 0)
    arange = torch.arange(T, device=halt_probs.device)
    prev = (N - 1).clamp(min=0)
    cum_before = torch.where(
        N > 0,
        cum[arange, prev],
        torch.zeros(T, dtype=cum.dtype, device=cum.device),
    )
    remainder = 1.0 - cum_before                   # R_t

    # weights: p_t(n) for n < N, R_t at n == N, 0 after
    positions = torch.arange(S, device=halt_probs.device).unsqueeze(0)  # (1, S)
    N_exp = N.unsqueeze(1)                                              # (T, 1)
    zeros = torch.zeros_like(halt_probs)

    weights = torch.where(
        positions < N_exp,
        halt_probs,
        torch.where(positions == N_exp, remainder.unsqueeze(1), zeros),
    )

    ponder = n_steps.to(remainder.dtype) + remainder
    return weights, n_steps.to(torch.int64), ponder


def act_output(states, weights):
    # states: (T, S, d), weights: (T, S) -> (T, d)
    return (weights.unsqueeze(-1) * states).sum(dim=1)