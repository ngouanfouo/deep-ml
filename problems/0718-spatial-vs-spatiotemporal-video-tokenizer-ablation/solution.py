import numpy as np

def video_tokenizer_reconstruct(video, patch_t: int, patch_h: int, patch_w: int, mode: str):
    """
    Reconstruct a video after mean-pool tokenization.

    Args:
        video: array-like of shape (T, H, W, C)
        patch_t: temporal patch size (ignored when mode == 'spatial')
        patch_h: spatial patch height
        patch_w: spatial patch width
        mode: 'spatial' or 'spatiotemporal'

    Returns:
        Reconstructed video as a nested list of shape (T, H, W, C).
    """
    video_arr = np.asarray(video)
    T, H, W, C = video_arr.shape
    
    # In 'spatial' mode, each frame is tokenized independently (patch_t forced to 1)
    pt = 1 if mode == 'spatial' else patch_t
    ph = patch_h
    pw = patch_w
    
    nT = T // pt
    nH = H // ph
    nW = W // pw
    
    # 1. Reshape video into non-overlapping patches
    v = video_arr.reshape(nT, pt, nH, ph, nW, pw, C)
    
    # 2. Transpose to group patch grid vs patch internal dimensions:
    #    Shape becomes (nT, nH, nW, pt, ph, pw, C)
    v_grid = v.transpose(0, 2, 4, 1, 3, 5, 6)
    
    # 3. Compute per-channel mean across all positions in each patch
    mean_val = np.mean(v_grid, axis=(3, 4, 5), keepdims=True)
    
    # 4. Broadcast the mean back to every position in the patch
    v_grid_recon = np.broadcast_to(mean_val, v_grid.shape)
    
    # 5. Undo transpose to restore layout (nT, pt, nH, ph, nW, pw, C)
    v_recon = v_grid_recon.transpose(0, 3, 1, 4, 2, 5, 6)
    
    # 6. Reshape back to original video shape (T, H, W, C)
    reconstructed_video = v_recon.reshape(T, H, W, C)
    
    return reconstructed_video.tolist()