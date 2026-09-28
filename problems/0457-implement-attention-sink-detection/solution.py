import numpy as np


def detect_attention_sinks(attn_weights: np.ndarray, threshold: float) -> dict:
    """
    Detect attention sink tokens from multi-head attention weight matrices.

    Args:
        attn_weights: Attention weights of shape (num_heads, seq_len, seq_len)
                      where attn_weights[h, i, j] is how much query i in head h
                      attends to key j.
        threshold: Minimum average received attention to qualify as a sink

    Returns:
        Dictionary with 'sink_positions', 'avg_attention_received', and 'sink_scores'
    """
    # Average across heads (axis 0) and queries (axis 1) → per key position
    avg_attention_received = attn_weights.mean(axis=(0, 1))   # (seq_len,)

    # Positions whose average received attention meets the threshold
    sink_positions = [
        int(j) for j in range(avg_attention_received.shape[0])
        if avg_attention_received[j] >= threshold
    ]

    # Full per-position averages, rounded
    avg_rounded = [round(float(v), 4) for v in avg_attention_received]

    # Scores for just the sink positions, in order
    sink_scores = [round(float(avg_attention_received[j]), 4) for j in sink_positions]

    return {
        "sink_positions": sink_positions,
        "avg_attention_received": avg_rounded,
        "sink_scores": sink_scores,
    }