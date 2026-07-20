import numpy as np

def sequential_video_diffusion(
    noise_chunks: np.ndarray,
    Wd: np.ndarray,
    alpha_bar: np.ndarray,
    n_context: int,
    n_overlap: int,
) -> np.ndarray:
    """
    Generate video frames sequentially using chunk-based diffusion denoising.

    Args:
        noise_chunks: Initial noise, shape (n_chunks, chunk_frames, H, W)
        Wd: Denoiser weight matrix, shape (H*W, H*W)
        alpha_bar: Cumulative noise schedule, shape (n_steps+1,)
        n_context: Number of context frames from previous chunk
        n_overlap: Number of frames to blend between consecutive chunks

    Returns:
        Generated video, shape (total_frames, H, W)
    """
    n_chunks, chunk_frames, H, W = noise_chunks.shape
    d_flat = H * W
    n_steps = len(alpha_bar) - 1
    
    # Store all independently generated chunks
    generated_chunks = []
    
    # Track the previous chunk's context frames to condition the current chunk
    prev_chunk_frames = None
    
    for c in range(n_chunks):
        # Initialize chunk frames with the initial noise configuration
        # shape: (chunk_frames, d_flat)
        current_chunk = noise_chunks[c].reshape(chunk_frames, d_flat)
        
        # Calculate context bias vector
        if c == 0 or prev_chunk_frames is None or n_context == 0:
            context_bias = np.zeros(d_flat)
        else:
            # Extract trailing context frames from the tail of the previous chunk
            context_frames = prev_chunk_frames[-n_context:]
            context_bias = np.mean(context_frames, axis=0)
            
        # Iterative deterministic DDIM-style denoising
        for t in range(n_steps, 0, -1):
            ab_t = alpha_bar[t]
            ab_prev = alpha_bar[t - 1]
            
            # Predict noise for each frame in the current chunk
            # eps_pred = Wd @ x + context_bias (vectorized across frames)
            eps_pred = np.matmul(current_chunk, Wd.T) + context_bias
            
            # Formulate the deterministic clean frame prediction (x0_pred)
            x0_pred = (current_chunk - np.sqrt(1.0 - ab_t) * eps_pred) / np.sqrt(ab_t)
            
            # Update to get the frame at the previous timestep (t-1)
            current_chunk = np.sqrt(ab_prev) * x0_pred + np.sqrt(1.0 - ab_prev) * eps_pred
            
        # Reshape frames back to spatial layout (chunk_frames, H, W)
        chunk_out = current_chunk.reshape(chunk_frames, H, W)
        generated_chunks.append(chunk_out)
        
        # Save the reference to the current chunk's flattened frames for the next loop
        prev_chunk_frames = current_chunk

    # --- Video Assembly and Boundary Blending ---
    if n_chunks == 1 or n_overlap == 0:
        return np.concatenate(generated_chunks, axis=0)
    
    # Compute total frames needed post-blending
    total_frames = chunk_frames + (n_chunks - 1) * (chunk_frames - n_overlap)
    assembled_video = np.zeros((total_frames, H, W))
    
    # Insert the first chunk completely
    assembled_video[:chunk_frames] = generated_chunks[0]
    current_end = chunk_frames
    
    for c in range(1, n_chunks):
        next_chunk = generated_chunks[c]
        overlap_start = current_end - n_overlap
        
        # 1. Linearly blend the overlapping boundary window
        for i in range(n_overlap):
            w_new = (i + 1) / (n_overlap + 1)
            w_old = 1.0 - w_new
            
            old_frame = assembled_video[overlap_start + i]
            new_frame = next_chunk[i]
            
            assembled_video[overlap_start + i] = w_old * old_frame + w_new * new_frame
            
        # 2. Append the remaining unique frames of the new chunk
        unique_frames_start = overlap_start + n_overlap
        unique_frames_end = unique_frames_start + (chunk_frames - n_overlap)
        assembled_video[unique_frames_start:unique_frames_end] = next_chunk[n_overlap:]
        
        current_end = unique_frames_end

    return assembled_video