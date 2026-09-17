import numpy as np

def tile_coding(
    state: list[float],
    num_tilings: int,
    tiles_per_dim: list[int],
    state_low: list[float],
    state_high: list[float]
) -> tuple[list[int], int]:
    """
    Compute active tile indices using tile coding.
    
    Args:
        state: Continuous state values (one per dimension)
        num_tilings: Number of overlapping tilings
        tiles_per_dim: Number of tiles per dimension in each tiling
        state_low: Lower bounds for each dimension
        state_high: Upper bounds for each dimension
        
    Returns:
        Tuple of (active_tile_indices, total_number_of_tiles)
    """
    D = len(state)
    # Tile width for each dimension
    tile_widths = [(state_high[d] - state_low[d]) / tiles_per_dim[d] for d in range(D)]
    # Total tiles per tiling (product of tiles per dimension)
    tiles_per_tiling = 1
    for s in tiles_per_dim:
        tiles_per_tiling *= s
    total_tiles = num_tilings * tiles_per_tiling

    active_tiles = []
    for t in range(num_tilings):
        idx_dims = []
        for d in range(D):
            # Adjusted position: (state - low) / width + t / num_tilings
            val = (state[d] - state_low[d]) / tile_widths[d] + t / num_tilings
            idx = int(np.floor(val))
            # Clip to valid range [0, tiles_per_dim[d] - 1]
            if idx < 0:
                idx = 0
            elif idx >= tiles_per_dim[d]:
                idx = tiles_per_dim[d] - 1
            idx_dims.append(idx)

        # Flatten row-major (C-style)
        local_index = 0
        for d in range(D):
            local_index = local_index * tiles_per_dim[d] + idx_dims[d]

        global_index = t * tiles_per_tiling + local_index
        active_tiles.append(global_index)

    return active_tiles, total_tiles