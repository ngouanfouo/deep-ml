import numpy as np

def vit_encode(image, patch_size: int, W_patch, cls_token, pos_embed, W_qkv, W_o, num_heads: int):
    """
    Forward pass of a minimal ViT encoder block.

    Returns the final embedding of the CLS token as a list of length D.
    """
    # Convert inputs to numpy arrays
    image = np.array(image, dtype=float)
    W_patch = np.array(W_patch, dtype=float)
    cls_token = np.array(cls_token, dtype=float)
    pos_embed = np.array(pos_embed, dtype=float)
    W_qkv = np.array(W_qkv, dtype=float)
    W_o = np.array(W_o, dtype=float)
    
    H, W, C = image.shape
    D = W_patch.shape[1]
    
    # Step 1: Extract patches and flatten them
    patches = []
    for i in range(0, H, patch_size):
        for j in range(0, W, patch_size):
            patch = image[i:i+patch_size, j:j+patch_size, :]
            patches.append(patch.flatten())  # C-order flatten
    patches = np.array(patches)  # shape (N, patch_size*patch_size*C)
    N = patches.shape[0]
    
    # Step 2: Project patches to D dimensions
    patch_embeds = patches @ W_patch  # shape (N, D)
    
    # Step 3: Prepend CLS token
    tokens = np.vstack([cls_token.reshape(1, D), patch_embeds])  # shape (N+1, D)
    
    # Step 4: Add positional embeddings
    tokens = tokens + pos_embed  # shape (N+1, D)
    
    # Step 5: Multi-head self-attention
    seq_len = tokens.shape[0]
    d_h = D // num_heads
    
    # Compute Q, K, V
    qkv = tokens @ W_qkv  # shape (N+1, 3D)
    Q = qkv[:, :D]        # shape (N+1, D)
    K = qkv[:, D:2*D]     # shape (N+1, D)
    V = qkv[:, 2*D:]      # shape (N+1, D)
    
    # Reshape for multi-head: (seq_len, num_heads, d_h) -> (num_heads, seq_len, d_h)
    Q = Q.reshape(seq_len, num_heads, d_h).transpose(1, 0, 2)
    K = K.reshape(seq_len, num_heads, d_h).transpose(1, 0, 2)
    V = V.reshape(seq_len, num_heads, d_h).transpose(1, 0, 2)
    
    # Scaled dot-product attention
    scale = 1.0 / np.sqrt(d_h)
    attn_scores = Q @ K.transpose(0, 2, 1) * scale  # shape (num_heads, seq_len, seq_len)
    
    # Softmax over last axis
    attn_weights = np.exp(attn_scores - np.max(attn_scores, axis=-1, keepdims=True))
    attn_weights = attn_weights / np.sum(attn_weights, axis=-1, keepdims=True)
    
    # Apply attention to values
    attn_output = attn_weights @ V  # shape (num_heads, seq_len, d_h)
    
    # Concatenate heads: transpose back and reshape
    attn_output = attn_output.transpose(1, 0, 2)  # shape (seq_len, num_heads, d_h)
    attn_output = attn_output.reshape(seq_len, D)  # shape (seq_len, D)
    
    # Step 6: Output projection
    output = attn_output @ W_o  # shape (seq_len, D)
    
    # Step 7: Return CLS token embedding (position 0)
    return output[0].tolist()