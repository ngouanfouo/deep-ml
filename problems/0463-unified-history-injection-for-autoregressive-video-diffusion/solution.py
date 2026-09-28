import numpy as np

def unified_history_injection(
    history_latents: np.ndarray,
    noisy_latents: np.ndarray,
    mode: str
) -> dict:
    """
    Pack clean history and noisy generation-target latents into a single
    unified input for an autoregressive video diffusion model.
    """
    T_h = history_latents.shape[0]
    T_n = noisy_latents.shape[0]

    # --- Representation control on the history -----------------------
    if mode == "t2v":
        # Zero out the entire history: generate purely from text.
        history_used = np.zeros_like(history_latents)
    elif mode == "i2v":
        # Keep only the last history frame; zero out earlier frames.
        history_used = np.zeros_like(history_latents)
        if T_h > 0:
            history_used[-1] = history_latents[-1]
    elif mode == "v2v":
        # Full history passed through unchanged.
        history_used = history_latents.copy()
    else:
        raise ValueError(
            f"mode must be 't2v', 'i2v', or 'v2v', got {mode!r}"
        )

    # --- Concatenate history first, then noisy targets ---------------
    unified_input = np.concatenate([history_used, noisy_latents], axis=0)

    # --- Context mask: 0 for history, 1 for noisy targets ------------
    context_mask = np.concatenate([
        np.zeros(T_h, dtype=int),
        np.ones(T_n, dtype=int),
    ])

    return {
        "unified_input": unified_input,
        "context_mask": context_mask,
        "num_history_frames": int(T_h),
        "num_noisy_frames": int(T_n),
    }