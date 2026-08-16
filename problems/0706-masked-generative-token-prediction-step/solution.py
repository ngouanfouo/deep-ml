import numpy as np
import math

def maskgit_decode_step(tokens, logits, step: int, total_steps: int, mask_id: int) -> np.ndarray:
    """
    Perform a single step of parallel masked token decoding.

    Args:
        tokens: 1D array-like of length N containing current token ids (mask_id for masked positions)
        logits: 2D array-like of shape (N, vocab_size) of model logits
        step: current step index (0-indexed)
        total_steps: total number of refinement steps
        mask_id: integer id used to indicate a masked position

    Returns:
        numpy array of length N with updated token ids.
    """
    tokens = np.array(tokens)
    logits = np.array(logits)
    N = len(tokens)

    # 1. Numerically stable softmax and model confidence calculation
    logits_max = np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    pred_tokens = np.argmax(probs, axis=-1)
    confidences = np.max(probs, axis=-1)

    # Identify initially masked positions
    is_masked = (tokens == mask_id)

    # 2. Fill currently masked positions with argmax predictions
    updated_tokens = tokens.copy()
    updated_tokens[is_masked] = pred_tokens[is_masked]

    # 3. Calculate number of positions that should remain masked using cosine schedule
    cos_val = math.cos((math.pi / 2.0) * (step + 1) / total_steps)
    num_remask = math.floor(N * cos_val)

    # Cap remaining mask count to available initially masked positions
    num_masked_count = int(np.sum(is_masked))
    num_remask = max(0, min(num_remask, num_masked_count))

    # 4. Re-mask positions with lowest confidence
    # Assign infinite confidence to already unmasked positions so they are never re-masked
    effective_conf = np.full(N, np.inf)
    effective_conf[is_masked] = confidences[is_masked]

    if num_remask > 0:
        # Select the lowest-confidence positions using numpy's default argsort
        remask_indices = np.argsort(effective_conf)[:num_remask]
        updated_tokens[remask_indices] = mask_id

    return updated_tokens