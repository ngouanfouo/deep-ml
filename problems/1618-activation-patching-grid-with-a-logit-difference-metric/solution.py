import numpy as np


def reader_metric(R, W_r, d_vec):
    """Attention from the last position over all positions, then d_vec . output."""
    R = np.asarray(R, dtype=float)
    W_r = np.asarray(W_r, dtype=float)
    d_vec = np.asarray(d_vec, dtype=float)

    P = R.shape[0]

    # Query is the residual at the last position.
    r_last = R[P - 1]

    # Scores s_p = r_last^T W_r r_p for every position p.
    scores = R @ (W_r.T @ r_last)  # (P,)

    # Stable softmax over positions.
    scores = scores - np.max(scores)
    exp_scores = np.exp(scores)
    attn = exp_scores / np.sum(exp_scores)

    # Attended mixture.
    output = attn @ R  # (d,)

    return float(d_vec @ output)


def assemble(embed, writes):
    embed = np.asarray(embed, dtype=float)
    writes = np.asarray(writes, dtype=float)
    return embed + writes.sum(axis=0)


def patching_grid(embed_clean, writes_clean, embed_corrupt, writes_corrupt, W_r, d_vec, decimals=4):
    """(C, P) normalised restoration scores."""
    embed_clean = np.asarray(embed_clean, dtype=float)
    writes_clean = np.asarray(writes_clean, dtype=float)
    embed_corrupt = np.asarray(embed_corrupt, dtype=float)
    writes_corrupt = np.asarray(writes_corrupt, dtype=float)

    C, P, _ = writes_clean.shape

    m_clean = reader_metric(assemble(embed_clean, writes_clean), W_r, d_vec)
    m_corrupt = reader_metric(assemble(embed_corrupt, writes_corrupt), W_r, d_vec)

    denom = m_clean - m_corrupt
    if denom == 0:
        return np.zeros((C, P), dtype=float)

    grid = np.zeros((C, P), dtype=float)
    for c in range(C):
        for p in range(P):
            patched_writes = writes_corrupt.copy()
            patched_writes[c, p] = writes_clean[c, p]
            m_patched = reader_metric(
                assemble(embed_corrupt, patched_writes), W_r, d_vec
            )
            grid[c, p] = (m_patched - m_corrupt) / denom

    return np.round(grid, decimals)


def best_patch(grid):
    grid = np.asarray(grid, dtype=float)
    flat_idx = int(np.argmax(grid))  # first occurrence in row-major order
    c, p = np.unravel_index(flat_idx, grid.shape)
    return int(c), int(p), float(grid[c, p])