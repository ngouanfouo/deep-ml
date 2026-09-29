import bisect

def progressive_batch_sizes(steps, milestones, token_budget):
    """
    Compute per-step batch sizes under a progressive context-length curriculum.

    Args:
        steps: list[int] of training step indices to query.
        milestones: list of [start_step, seq_len], sorted ascending by start_step,
                    with the first entry having start_step == 0.
        token_budget: int, target tokens per batch.

    Returns:
        list[int]: batch size at each queried step.
    """
    start_steps = [m[0] for m in milestones]
    seq_lens = [m[1] for m in milestones]

    batch_sizes = []
    for s in steps:
        idx = bisect.bisect_right(start_steps, s) - 1
        L = seq_lens[idx]
        B = token_budget // L
        if B < 1:
            B = 1
        batch_sizes.append(B)

    return batch_sizes