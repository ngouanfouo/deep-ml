import numpy as np

def compute_perceptual_distance(features_ref, features_gen, channel_weights):
    eps = 1e-10
    total = 0.0
    
    for f_ref, f_gen, w in zip(features_ref, features_gen, channel_weights):
        f_ref = np.asarray(f_ref, dtype=float)
        f_gen = np.asarray(f_gen, dtype=float)
        w = np.asarray(w, dtype=float)
        
        # L2-normalize the channel vector at each spatial location
        norm_ref = np.sqrt(np.sum(f_ref ** 2, axis=0, keepdims=True)) + eps
        norm_gen = np.sqrt(np.sum(f_gen ** 2, axis=0, keepdims=True)) + eps
        f_ref_n = f_ref / norm_ref
        f_gen_n = f_gen / norm_gen
        
        # Squared difference between normalized features
        diff_sq = (f_ref_n - f_gen_n) ** 2
        
        # Apply per-channel weights
        w_shape = (w.shape[0],) + (1,) * (f_ref.ndim - 1)
        weighted = diff_sq * w.reshape(w_shape)
        
        # Sum over channels, then average over spatial locations
        per_spatial = np.sum(weighted, axis=0)   # shape (H, W)
        total += np.mean(per_spatial)
    
    return float(total)