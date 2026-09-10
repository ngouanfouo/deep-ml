import numpy as np

def clip_score_filter(image_embeds, text_embeds, threshold: float):
    """
    Compute pairwise cosine similarities between image and text embeddings
    (assumed L2-normalized) and zero out entries below `threshold`.

    Returns:
        Nested list of shape (n_images, n_texts).
    """
    # Convert to numpy arrays
    image_embeds = np.array(image_embeds)
    text_embeds = np.array(text_embeds)
    
    # Check for zero rows
    if image_embeds.size == 0 or text_embeds.size == 0:
        return []
    if image_embeds.shape[0] == 0 or text_embeds.shape[0] == 0:
        return []
    
    # Compute cosine similarity matrix (dot product since L2-normalized)
    similarity_matrix = image_embeds @ text_embeds.T
    
    # Zero out entries strictly below threshold
    similarity_matrix[similarity_matrix < threshold] = 0.0
    
    # Convert to nested Python list
    return similarity_matrix.tolist()