import numpy as np

def guidance_attention_mask(
    chunk_sizes: list[int],
    current_chunk: int
) -> np.ndarray:
    """
    Build a boolean attention mask for chunked autoregressive video generation.

    Args:
        chunk_sizes:   tokens per chunk, in order
        current_chunk: index of the chunk being generated

    Returns:
        Boolean array of shape (total_tokens, total_tokens).
        True means the row token can attend to the column token.
    """
    total_tokens = sum(chunk_sizes)
    mask = np.zeros((total_tokens, total_tokens), dtype=bool)

    # Start index of the current chunk
    start_current = sum(chunk_sizes[:current_chunk])

    # 1) Historical tokens: attend to all historical tokens (bidirectional).
    #    This includes themselves and each other, but NOT the current chunk.
    mask[:start_current, :start_current] = True

    # 2) Current chunk tokens: full access to history, causal within the chunk.
    #    Row i (global index) attends to columns 0 .. i (inclusive).
    for i in range(start_current, total_tokens):
        mask[i, :i + 1] = True

    return mask