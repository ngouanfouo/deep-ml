import numpy as np


def compress_blocks(kv, seq_ids, m):
    """
    Compress a contiguous token array kv (n, c) per sequence: for each sequence id in order of
    first appearance, chunk its rows into blocks of m (drop the trailing remainder) and emit
    the mean of each block. Returns an array of shape (num_blocks, c).
    """
    kv = np.asarray(kv)
    seq_ids = np.asarray(seq_ids)

    n = kv.shape[0]
    c = kv.shape[1] if kv.ndim > 1 else 0

    if n == 0:
        return np.zeros((0, c), dtype=float)

    blocks = []
    i = 0
    while i < n:
        sid = seq_ids[i]
        j = i
        while j < n and seq_ids[j] == sid:
            j += 1

        length = j - i
        n_blocks = length // m
        for b in range(n_blocks):
            start = i + b * m
            blocks.append(kv[start:start + m].mean(axis=0))

        i = j

    if not blocks:
        return np.zeros((0, c), dtype=float)
    return np.vstack(blocks)


def cp_two_stage(shards, seq_ids, m):
    """
    Emulate the two-stage CP communication of report section 3.4.3.
    shards: list of cp_size arrays (s, c) in rank order; seq_ids: labels for all cp_size*s tokens.
    Stage one: rank r receives the last m rows of rank r-1 and compresses the blocks whose last
    token is local, padding with zero rows to s//m + 1 rows.
    Stage two: all-gather, keep the valid rows in rank order, pad with zeros to cp_size*(s//m).
    Returns {"compressed": (cp_size*s//m, c) array, "valid": list of bools, "per_rank_counts": list}.
    """
    shards = [np.asarray(sh, dtype=float) for sh in shards]

    if len(shards) == 0:
        return {"compressed": np.zeros((0, 0)), "valid": [], "per_rank_counts": []}

    cp_size = len(shards)
    s, c = shards[0].shape
    kv = np.concatenate(shards, axis=0)
    seq_ids = np.asarray(seq_ids)

    n_global = cp_size * s

    # Mark every global position that ends a full block of m tokens
    # within its own (contiguous) sequence.
    block_end = np.zeros(n_global, dtype=bool)
    run_start = 0
    for p in range(n_global):
        if p > 0 and seq_ids[p] != seq_ids[p - 1]:
            run_start = p
        offset = p - run_start
        if (offset + 1) % m == 0:
            block_end[p] = True

    slot_per_rank = s // m + 1
    rank_outputs = []
    per_rank_counts = []

    for r in range(cp_size):
        start = r * s
        end = start + s

        blocks = []
        for p in range(start, end):
            if block_end[p]:
                blocks.append(kv[p - m + 1:p + 1].mean(axis=0))

        count = len(blocks)
        per_rank_counts.append(count)

        out = np.zeros((slot_per_rank, c), dtype=float)
        if count:
            out[:count] = np.vstack(blocks)
        rank_outputs.append(out)

    # Stage two: keep real rows in rank order, then pad to the final slot count.
    real_rows = []
    for r, count in enumerate(per_rank_counts):
        if count > 0:
            real_rows.append(rank_outputs[r][:count])

    if real_rows:
        real = np.vstack(real_rows)
    else:
        real = np.zeros((0, c), dtype=float)

    total_slots = cp_size * (s // m)
    real = real[:total_slots]
    n_real = real.shape[0]

    compressed = np.zeros((total_slots, c), dtype=float)
    compressed[:n_real] = real
    valid = [True] * n_real + [False] * (total_slots - n_real)

    return {
        "compressed": compressed,
        "valid": valid,
        "per_rank_counts": per_rank_counts,
    }