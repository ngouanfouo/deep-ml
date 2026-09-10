import numpy as np

def patch_embed_reconstruct(image: np.ndarray, patch_size: int):
    """
    Split image into patches, flatten to patch embedding matrix,
    then reconstruct the original image.

    Args:
        image: 2D numpy array of shape (H, W)
        patch_size: int, side length of each square patch

    Returns:
        Reconstructed image as a nested list of shape (H, W),
        or -1 if dimensions are invalid.
    """
    image = np.array(image)
    H, W = image.shape
    
    # Check divisibility
    if H % patch_size != 0 or W % patch_size != 0:
        return -1
    
    n_rows = H // patch_size
    n_cols = W // patch_size
    N = n_rows * n_cols
    
    # Step 1: Extract patches and flatten each into a row -> (N, p*p)
    patches = image.reshape(n_rows, patch_size, n_cols, patch_size)
    patches = patches.transpose(0, 2, 1, 3)      # (n_rows, n_cols, p, p)
    embed_matrix = patches.reshape(N, patch_size * patch_size)
    
    # Step 2: Reconstruct the original image from the embedding matrix
    reconstructed = embed_matrix.reshape(n_rows, n_cols, patch_size, patch_size)
    reconstructed = reconstructed.transpose(0, 2, 1, 3)  # (n_rows, p, n_cols, p)
    reconstructed = reconstructed.reshape(H, W)
    
    return reconstructed.tolist()