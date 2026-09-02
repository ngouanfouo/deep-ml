import numpy as np

def per_layer_embedding(input_ids, token_embed, ple_embed, proj_weight, rms_weight, d_ple, n_layers, eps=1e-6):
    """
    Compute per-layer embeddings by fusing a projected main embedding
    (after RMSNorm) with a scaled per-layer embedding lookup.

    Returns a nested Python list of shape (B, T, n_layers, d_ple).
    """
    input_ids = np.array(input_ids, dtype=int)
    token_embed = np.array(token_embed, dtype=float)
    ple_embed = np.array(ple_embed, dtype=float)
    proj_weight = np.array(proj_weight, dtype=float)
    rms_weight = np.array(rms_weight, dtype=float)
    
    B, T = input_ids.shape
    
    # 1. Look up the main token embedding: main = token_embed[input_ids] with shape (B, T, d_model)
    main = token_embed[input_ids]
    
    # 2. Look up the per-layer embedding and multiply by sqrt(d_ple)
    ple = ple_embed[input_ids] * np.sqrt(d_ple)
    
    # 3. Project the main embedding to per-layer space: proj = main @ proj_weight
    proj = main @ proj_weight
    
    # 4. Apply RMSNorm to proj along the last axis, then elementwise multiply by rms_weight
    # RMSNorm(x) = x / sqrt(mean(x^2) + eps)
    variance = np.mean(proj ** 2, axis=-1, keepdims=True)
    proj_norm = (proj / np.sqrt(variance + eps)) * rms_weight
    
    # 5. Combine the normalized projection and the scaled PLE
    combined = (proj_norm + ple) * (2 ** -0.5)
    
    # 6. Reshape combined to (B, T, n_layers, d_ple)
    combined_reshaped = combined.reshape(B, T, n_layers, d_ple)
    
    # Return as a nested Python list
    return combined_reshaped.tolist()