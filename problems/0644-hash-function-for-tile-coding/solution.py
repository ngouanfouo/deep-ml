import numpy as np

def tile_coding_hash(state: list, num_tilings: int, tiles_per_dim: int, memory_size: int, state_bounds: list = None) -> list:
    """
    Compute active tile indices using tile coding with hash-based index mapping.

    Args:
        state: list of floats, continuous state variables.
        num_tilings: int, number of tilings.
        tiles_per_dim: int, tiles per dimension per tiling.
        memory_size: int, total hash table size.
        state_bounds: list of (low, high) tuples per dimension (default: (0,1)).

    Returns:
        List of int, one active tile index per tiling.
    """
    state = np.asarray(state, dtype=float)
    D = state.shape[0]
    
    # Default bounds: (0, 1) for each dimension
    if state_bounds is None:
        state_bounds = [(0.0, 1.0)] * D
    
    # Precompute the base scaled coordinate for each dimension
    # scaled[d] = (state[d] - low) / (high - low) * tiles_per_dim
    base = np.zeros(D, dtype=float)
    for d in range(D):
        low, high = state_bounds[d]
        base[d] = (state[d] - low) / (high - low) * tiles_per_dim
    
    active_tiles = []
    for t in range(num_tilings):
        offset = t / num_tilings
        
        # Build coordinate tuple: (tiling_index, tile_coord_0, tile_coord_1, ...)
        coords = [t]
        for d in range(D):
            shifted = base[d] + offset
            coords.append(int(np.floor(shifted)))
        
        # djb2 hash with 32-bit masking
        h = 5381
        for c in coords:
            h = ((h * 33) + c) & 0xFFFFFFFF
        
        active_tiles.append(h % memory_size)
    
    return active_tiles