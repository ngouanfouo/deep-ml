import numpy as np

def video_tokenize(video, codebook, encoder_weight, patch_size):
    """
    Tokenize a video into a grid of discrete codebook indices using a
    spatiotemporal VQ-VAE encoder.

    Args:
        video: array of shape (T, H, W, C).
        codebook: array of shape (K, D).
        encoder_weight: array of shape (pt*ph*pw*C, D).
        patch_size: tuple (pt, ph, pw).

    Returns:
        Nested list of shape (T/pt, H/ph, W/pw) with codebook indices,
        or -1 if the inputs are invalid.
    """
    try:
        video = np.asarray(video)
        codebook = np.asarray(codebook)
        encoder_weight = np.asarray(encoder_weight)

        # Dimension and rank validation
        if video.ndim != 4 or codebook.ndim != 2 or encoder_weight.ndim != 2:
            return -1

        if not (isinstance(patch_size, (tuple, list, np.ndarray)) and len(patch_size) == 3):
            return -1

        pt, ph, pw = patch_size
        if pt != int(pt) or ph != int(ph) or pw != int(pw):
            return -1
        pt, ph, pw = int(pt), int(ph), int(pw)

        if pt <= 0 or ph <= 0 or pw <= 0:
            return -1

        T, H, W, C = video.shape
        K, D = codebook.shape

        if T <= 0 or H <= 0 or W <= 0 or C <= 0 or K <= 0 or D <= 0:
            return -1

        # Check exact divisibility
        if T % pt != 0 or H % ph != 0 or W % pw != 0:
            return -1

        # Validate projection matrix dimensions
        expected_in_dim = pt * ph * pw * C
        if encoder_weight.shape[0] != expected_in_dim or encoder_weight.shape[1] != D:
            return -1

    except Exception:
        return -1

    nT, nH, nW = T // pt, H // ph, W // pw

    # 1. Extract non-overlapping spatiotemporal patches
    v = video.reshape(nT, pt, nH, ph, nW, pw, C)
    v = v.transpose(0, 2, 4, 1, 3, 5, 6)  # Shape: (nT, nH, nW, pt, ph, pw, C)
    patches_flat = v.reshape(nT, nH, nW, expected_in_dim)

    # 2. Linear projection into latent space
    latents = patches_flat @ encoder_weight  # Shape: (nT, nH, nW, D)
    latents_flat = latents.reshape(-1, D)

    # 3. Vector Quantization via nearest codebook lookup
    diff = latents_flat[:, np.newaxis, :] - codebook[np.newaxis, :, :]  # Shape: (N_patches, K, D)
    dists = np.sum(diff ** 2, axis=-1)  # Squared Euclidean distance

    indices = np.argmin(dists, axis=1)  # Automatically breaks ties with lowest index
    grid_indices = indices.reshape(nT, nH, nW)

    return grid_indices.tolist()