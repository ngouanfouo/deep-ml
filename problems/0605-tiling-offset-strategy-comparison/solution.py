import numpy as np

def compare_tiling_offsets(
    states: list,
    num_tilings: int,
    tiles_per_dim: list,
    state_low: list,
    state_high: list,
    strategy: str
) -> tuple:
    states = np.asarray(states, dtype=float)
    n_states, n_dims = states.shape
    tiles_per_dim = list(tiles_per_dim)
    low = np.asarray(state_low, dtype=float)
    high = np.asarray(state_high, dtype=float)

    tile_width = (high - low) / np.asarray(tiles_per_dim, dtype=float)
    tiles_per_tiling = int(np.prod(tiles_per_dim))

    # Precompute the shift of each tiling in each dimension
    shifts = np.zeros((num_tilings, n_dims))
    for t in range(num_tilings):
        for d in range(n_dims):
            unit = 1 if strategy == "uniform" else (2 * d + 1)
            shifts[t, d] = t * unit * tile_width[d] / num_tilings

    active_tiles = []
    for s in states:
        tiles = []
        for t in range(num_tilings):
            flat = 0
            for d in range(n_dims):
                adjusted = (s[d] - low[d] + shifts[t, d]) / tile_width[d]
                idx = int(np.floor(adjusted))
                idx = max(0, min(idx, tiles_per_dim[d] - 1))
                flat = flat * tiles_per_dim[d] + idx       # row-major flatten
            tiles.append(flat + t * tiles_per_tiling)      # global index
        active_tiles.append(tiles)

    # Pairwise similarity: fraction of shared active tiles
    similarity = []
    for i in range(n_states):
        set_i = set(active_tiles[i])
        row = []
        for j in range(n_states):
            shared = len(set_i & set(active_tiles[j]))
            row.append(round(shared / num_tilings, 4))
        similarity.append(row)

    return active_tiles, similarity