import numpy as np

def tile_image(image, tile_size, max_tiles):
    """
    Tile an image into (n_tiles, T, T, C) with an aspect-ratio preserving canvas.

    Args:
        image: array-like of shape (H, W, C)
        tile_size: int, side length T of each square tile
        max_tiles: int, maximum allowed n_h * n_w

    Returns:
        Nested list of shape (n_h*n_w, T, T, C).
    """
    image = np.asarray(image)
    H, W, C = image.shape
    T = tile_size
    
    # --- Enumerate every tiling inside the budget ---
    candidates = []
    for n_h in range(1, max_tiles + 1):
        for n_w in range(1, max_tiles + 1):
            if n_h * n_w <= max_tiles:
                err = abs(n_h / n_w - H / W)
                candidates.append((err, n_h * n_w, n_h, n_w))
    
    # Order by aspect-ratio error, then fewer total tiles, then smaller n_h
    candidates.sort()
    _, _, n_h, n_w = candidates[0]
    
    new_H = n_h * T
    new_W = n_w * T
    
    # --- Nearest-neighbor resize (exact integer floor) ---
    row_idx = (np.arange(new_H) * H) // new_H   # floor(i * H / new_H)
    col_idx = (np.arange(new_W) * W) // new_W   # floor(j * W / new_W)
    
    resized = image[np.ix_(row_idx, col_idx)]   # (new_H, new_W, C)
    
    # --- Split into n_h x n_w tiles of (T, T, C) in row-major order ---
    # reshape new_H = n_h*T as (n_h, T) and new_W = n_w*T as (n_w, T)
    tiles = resized.reshape(n_h, T, n_w, T, C)
    tiles = tiles.transpose(0, 2, 1, 3, 4)      # (n_h, n_w, T, T, C)
    tiles = tiles.reshape(n_h * n_w, T, T, C)   # flatten tile grid row-major
    
    return tiles.tolist()